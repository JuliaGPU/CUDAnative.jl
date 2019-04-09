import BenchmarkTools

function should_use_gc()
    try
        return use_gc
    catch ex
        return true
    end
end

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
    esc(quote
        if should_use_gc()
            CUDAnative.@cuda_gc $(args...)
        else
            @sync CUDAnative.@cuda $(args...)
        end
    end)
end

suite = BenchmarkTools.BenchmarkGroup()

function register_cuda_benchmark(f, name, config)
    suite[name][config] = BenchmarkTools.@benchmarkable $f() setup=(set_malloc_heap_size(BENCHMARK_HEAP_SIZE); $f()) teardown=(device_reset!()) evals=1 seconds=90
end

macro cuda_benchmark(name, ex)
    esc(quote
        suite[$name] = BenchmarkTools.BenchmarkGroup(["gc", "nogc"])
        register_cuda_benchmark($name, "gc") do
            global use_gc = true
            $(ex)
        end
        register_cuda_benchmark($name, "nogc") do
            global use_gc = false
            $(ex)
        end
    end)
end

function run_benchmarks()
    BenchmarkTools.run(suite)
end
