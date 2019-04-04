using StaticArrays, CUDAnative, CUDAdrv

include("utils.jl")

# This benchmark makes every thread allocate a large matrix.
# It stresses the allocator's ability to quickly allocate
# a small number of very large objects.

const matrix_dim = 40
const thread_count = 256

function fill()
    m = zeros(MMatrix{matrix_dim, matrix_dim, Int64})

    for i in 1:matrix_dim
        for j in 1:matrix_dim
            m[i, j] = i * j
        end
    end

    return m
end

function kernel(result::CUDAnative.DevicePtr{Int64})
    thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    unsafe_store!(result, fill()[20, 30], thread_id)
    return
end

function benchmark()
    destination_array = Mem.alloc(Int64, thread_count)
    destination_pointer = Base.unsafe_convert(CuPtr{Int64}, destination_array)
    @cuda_sync threads=thread_count kernel(destination_pointer)
end

@cuda_benchmark benchmark()
