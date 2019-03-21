using StaticArrays, CUDAnative, CUDAdrv

use_gc = false

const matrix_dim = 40
const iterations = 20
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

    # Write the accumulator to the result array.
    unsafe_store!(result, fill()[20, 30], thread_id)

    return
end

destination_array = Mem.alloc(Int64, thread_count)
destination_pointer = Base.unsafe_convert(CuPtr{Int64}, destination_array)

if use_gc
    @cuda_gc threads=thread_count kernel(destination_pointer)
    stats = @cuda_gc threads=thread_count kernel(destination_pointer)
else
    @cuda threads=thread_count kernel(destination_pointer)
    stats = CUDAdrv.@elapsed @cuda threads=thread_count kernel(destination_pointer)
end
println(stats)
