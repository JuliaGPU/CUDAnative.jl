import BenchmarkTools

use_gc = true

const MiB = 1 << 20
const CU_LIMIT_MALLOC_HEAP_SIZE = 0x02
const BENCHMARK_HEAP_SIZE = 64 * MiB

function set_malloc_heap_size(size::Integer)
    CUDAdrv.@apicall(
        :cuCtxSetLimit,
        (Cint, Csize_t),
        CU_LIMIT_MALLOC_HEAP_SIZE,
        Csize_t(size))
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
        local stats = BenchmarkTools.@benchmark $(ex) setup=(set_malloc_heap_size(BENCHMARK_HEAP_SIZE); $(ex)) teardown=(device_reset!()) evals=1
        println(length(stats))
        println(stats)
    end)
end
