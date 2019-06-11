module ArrayFeatures

using CUDAdrv, CUDAnative

# This benchmark has every thread exercise the core low-level
# array API.

const thread_count = 256

# Creates an array of Fibonacci numbers.
function fib_array(count::Integer)
    # Calls `jl_alloc_array_1d`.
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

function iterative_sum(array)
    result = 0
    for i in array
        result += i
    end
    return result
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

    # Create a two-dimensional array (calls `jl_alloc_array_2d`).
    arr_2d = fill(2, (2, 2))

    # Create a three-dimensional array (calls `jl_alloc_array_3d`).
    arr_3d = fill(2, (2, 2, 2))

    # Create a four-dimensional array (calls `jl_new_array`).
    arr_4d = fill(2, (2, 2, 2, 2))

    # Create an alias for the Fibonacci array (this is dangerous, but we
    # know what we're doing here; calls `jl_ptr_to_array_1d`).
    alias = unsafe_wrap(Array, pointer(arr), length(arr))

    # Create an alias for `arr_2d` (calls `jl_ptr_to_array`).
    alias_2d = unsafe_wrap(Array, pointer(arr_2d), size(arr_2d))

    # Create an array that is similar to `arr_3d` and fill it with constants.
    # This does not call any new low-level functions, but it does illustrate
    # that high-level functions such as `similar` and `fill!` fully functional.
    arr_3d_sim = similar(arr_3d)
    fill!(arr_3d_sim, 10)

    return iterative_sum(arr) +
        iterative_sum(arr_2d) +
        iterative_sum(arr_3d) +
        iterative_sum(arr_4d) +
        iterative_sum(alias) +
        iterative_sum(alias_2d) +
        iterative_sum(arr_3d_sim)
end

function kernel(destination)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    for j in 1:3
        unsafe_store!(destination, manipulate_array(), i)
    end
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
