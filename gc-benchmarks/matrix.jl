module Matrix

using StaticArrays, CUDAnative, CUDAdrv

# This benchmark makes every thread allocate a large matrix.
# It stresses the allocator's ability to quickly allocate
# very large objects.

const matrix_dim = 40
const thread_count = 256

@noinline function escape(value)
    Base.pointer_from_objref(value)
    value
end

function fill()
    m = zeros(MMatrix{matrix_dim, matrix_dim, Int64})

    for i in 1:matrix_dim
        for j in 1:matrix_dim
            m[i, j] = i * j
        end
    end

    return escape(m)
end

function kernel(result::CUDAnative.DevicePtr{Int64})
    thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    for i in 1:6
        unsafe_store!(result, fill()[20, 30], thread_id)
    end
    return
end

end

function matrix_benchmark()
    destination_array = Mem.alloc(Mem.DeviceBuffer, sizeof(Int64) * Matrix.thread_count)
    destination_pointer = Base.unsafe_convert(CuPtr{Int64}, destination_array)
    @cuda_sync threads=Matrix.thread_count Matrix.kernel(destination_pointer)
end

@cuda_benchmark "matrix" matrix_benchmark()
