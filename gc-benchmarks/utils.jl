import BenchmarkTools

use_gc = true

"""
    device_reset!(dev::CuDevice=device())

Reset the CUDA state associated with a device. This call with release the underlying
context, at which point any objects allocated in that context will be invalidated.
"""
function device_reset!(dev::CuDevice=CUDAdrv.device())
    delete!(CUDAnative.device_contexts, dev)

    pctx = CuPrimaryContext(dev)
    unsafe_reset!(pctx)

    # unless the user switches devices, new API calls should trigger initialization
    CUDAdrv.apicall_hook[] = CUDAnative.maybe_initialize
    CUDAnative.initialized[] = false

    # HACK: primary contexts always have the same handle, defeating the compilation cache
    empty!(CUDAnative.compilecache)
end

"""
    @sync ex
Run expression `ex` and synchronize the GPU afterwards. This is a CPU-friendly
synchronization, i.e. it performs a blocking synchronization without increasing CPU load. As
such, this operation is preferred over implicit synchronization (e.g. when performing a
memory copy) for high-performance applications.
It is also useful for timing code that executes asynchronously.
"""
macro sync(ex)
    # Copied from https://github.com/JuliaGPU/CuArrays.jl/blob/8e45a27f2b12796f47683340845f98f017865676/src/utils.jl#L68-L86
    quote
        local e = CuEvent(CUDAdrv.EVENT_BLOCKING_SYNC | CUDAdrv.EVENT_DISABLE_TIMING)
        local ret = $(esc(ex))
        CUDAdrv.record(e)
        CUDAdrv.synchronize(e)
        ret
    end
end

macro cuda_sync(args...)
    if use_gc
        esc(quote
            CUDAnative.@cuda_gc $(args...)
        end)
    else
        esc(quote
            @sync CUDAnative.@cuda $(args...)
        end)
    end
end

macro cuda_benchmark(ex)
    esc(quote
        local stats = BenchmarkTools.@benchmark $(ex) setup=($(ex)) teardown=(device_reset!()) evals=1
        println(length(stats))
        println(stats)
    end)
end
