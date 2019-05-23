import BenchmarkTools

function get_gc_mode()
    try
        return gc_mode
    catch ex
        return "gc"
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
        local mode = get_gc_mode()
        if mode == "gc"
            CUDAnative.@cuda gc=true gc_config=gc_config $(args...)
        elseif mode == "bump"
            local capacity = 60 * MiB
            local buf = Mem.alloc(Mem.DeviceBuffer, capacity)
            local start_address = pointer(buf)
            local function init(kernel)
                CUDAnative.Runtime.bump_alloc_init!(kernel, start_address, capacity)
            end
            @sync CUDAnative.@cuda init=init malloc="ptx_bump_alloc" $(args...)
            Mem.free(buf)
        else
            @sync CUDAnative.@cuda $(args...)
        end
    end)
end

suite = BenchmarkTools.BenchmarkGroup()

function register_cuda_benchmark(f, name, config)
    suite[name][config] = BenchmarkTools.@benchmarkable $f() setup=(set_malloc_heap_size(BENCHMARK_HEAP_SIZE); $f()) teardown=(device_reset!()) evals=1
end

const MiB = 1 << 20

macro cuda_benchmark(name, ex)
    esc(quote
        suite[$name] = BenchmarkTools.BenchmarkGroup(["gc", "gc-shared", "nogc", "bump"])
        register_cuda_benchmark($name, "gc") do
            global gc_mode = "gc"
            global gc_config = GCConfiguration(local_arena_count=8, local_arena_initial_size=MiB, global_arena_initial_size=2 * MiB)
            $(ex)
        end
        register_cuda_benchmark($name, "gc-shared") do
            global gc_mode = "gc"
            global gc_config = GCConfiguration(local_arena_count=0, global_arena_initial_size=10 * MiB)
            $(ex)
        end
        register_cuda_benchmark($name, "nogc") do
            global gc_mode = "nogc"
            $(ex)
        end
        register_cuda_benchmark($name, "bump") do
            global gc_mode = "bump"
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

function upload!(destination, source)
    Mem.copy!(destination, pointer(source), sizeof(source))
end

function download(::Type{T}, source, dims) where T
    result = Array{T}(undef, dims)
    Mem.copy!(pointer(result), source, sizeof(result))
    result
end
