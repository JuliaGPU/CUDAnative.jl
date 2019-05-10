module ArrayExpansion

using CUDAdrv, CUDAnative

# This benchmark has every thread create arrays and repeatedly
# append elements to those arrays.

const thread_count = 256
const array_length = 200
const runs = 10

function iterative_sum(elements::Array{T})::T where T
    result = zero(T)
    for i in elements
        result += i
    end
    return result
end

function kernel(destination)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    result = 0
    for j in 1:runs
        array = Int[]
        for k in 1:array_length
            push!(array, k)
        end
        result += iterative_sum(array)
    end
    unsafe_store!(destination, result, i)
    return
end

end

function array_expansion_benchmark()
    destination_array = Mem.alloc(Mem.DeviceBuffer, sizeof(Int) * ArrayExpansion.thread_count)
    destination_pointer = Base.unsafe_convert(CuPtr{Int}, destination_array)

    # Run the kernel.
    @cuda_sync threads=ArrayExpansion.thread_count ArrayExpansion.kernel(destination_pointer)

    @test download(Int, destination_array, ArrayExpansion.thread_count) == fill(ArrayExpansion.runs * sum(1:ArrayExpansion.array_length), ArrayExpansion.thread_count)
end

@cuda_benchmark "array expansion" array_expansion_benchmark()
