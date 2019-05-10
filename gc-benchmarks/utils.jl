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
            CUDAnative.@cuda_gc gc_config=gc_config $(args...)
        else
            @sync CUDAnative.@cuda $(args...)
        end
    end)
end

suite = BenchmarkTools.BenchmarkGroup()

function register_cuda_benchmark(f, name, config)
    suite[name][config] = BenchmarkTools.@benchmarkable $f() setup=(set_malloc_heap_size(BENCHMARK_HEAP_SIZE); $f()) teardown=(device_reset!()) evals=1 seconds=90
end

const MiB = 1 << 20

macro cuda_benchmark(name, ex)
    esc(quote
        suite[$name] = BenchmarkTools.BenchmarkGroup(["gc", "gc-shared", "nogc"])
        register_cuda_benchmark($name, "gc") do
            global use_gc = true
            global gc_config = GCConfiguration(local_arena_count=8, local_arena_initial_size=MiB, global_arena_initial_size=2 * MiB)
            $(ex)
        end
        register_cuda_benchmark($name, "gc-shared") do
            global use_gc = true
            global gc_config = GCConfiguration(local_arena_count=0, global_arena_initial_size=10 * MiB)
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

module CUDArandom

# A linear congruential pseudo-random number generator.
mutable struct LinearCongruentialGenerator
    modulus::Int
    a::Int
    c::Int
    state::Int
end

LinearCongruentialGenerator(seed::Int) = LinearCongruentialGenerator(1 << 32, 1664525, 1013904223, seed)

# Requests a pseudo-random number.
function next(generator::LinearCongruentialGenerator)::Int
    generator.state = (generator.a * generator.state + generator.c) % generator.modulus
    generator.state
end

# Requests a pseudo-random number that is at least as great as `lower`
# and less than `upper`.
function next(generator::LinearCongruentialGenerator, lower::Int, upper::Int)::Int
    lower + next(generator) % (upper - lower)
end

end
