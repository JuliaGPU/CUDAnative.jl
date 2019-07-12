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
