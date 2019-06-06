module ArrayFeatures

using CUDAdrv, CUDAnative

# This benchmark has every thread exercise the entire low-level
# array API.

const thread_count = 256

# Creates an array of Fibonacci numbers.
function fib_array(count::Integer)
    result = [1, 1]
    # Calls `jl_array_sizehint`.
    sizehint!(result, count + 2)
    for i in 1:count
        # Calls `jl_array_grow_end`.
        push!(result, result[i] + result[i + 1])
    end
    return result
end

function intersperse_with!(vec::Vector{T}, value::T) where T
    for i in 1:length(vec)
        # Calls `jl_array_grow_at`.
        insert!(vec, i * 2, value)
    end
    return vec
end

function manipulate_array()
    # Initialize the array as a Fibonacci sequence.
    arr = fib_array(20)

    # Intersperse the array with constants.
    intersperse_with!(arr, 2)

    # Prepend a constant to the array (calls `jl_array_grow_beg`).
    pushfirst!(arr, 2)

    # Intersperse again.
    intersperse_with!(arr, 4)

    # Delete the first element (calls `jl_array_del_beg`).
    popfirst!(arr)

    # Delete the last element (calls `jl_array_del_end`).
    pop!(arr)

    # Delete some other element (calls `jl_array_del_at`).
    deleteat!(arr, 8)

    result = 0
    for i in arr
        result += i
    end
    return result
end

function kernel(destination)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    unsafe_store!(destination, manipulate_array(), i)
    return
end

end

function array_features_benchmark()
    destination_array = Mem.alloc(Mem.DeviceBuffer, sizeof(Int) * ArrayFeatures.thread_count)
    destination_pointer = Base.unsafe_convert(CuPtr{Int}, destination_array)

    # Run the kernel.
    @cuda_sync threads=ArrayFeatures.thread_count ArrayFeatures.kernel(destination_pointer)

    @test download(Int, destination_array, ArrayFeatures.thread_count) == fill(ArrayFeatures.manipulate_array(), ArrayFeatures.thread_count)
end

@cuda_benchmark "array features" array_features_benchmark()
