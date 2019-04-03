# This example has kernels allocate dense symmetric matrices, fill them with Fibonacci numbers
# and compute their squares. The example is designed to stress the garbage allocator, specifically
# testing its ability to deal with many large objects. Furthermore, the example requires multiple
# collections to run to completion, so it also tests the performance of those collections.

using StaticArrays, CUDAnative, CUDAdrv
import Base: getindex, setindex!, pointer, unsafe_convert, zeros

const use_gc = true

"""A fixed-size, heap-allocated array type for CUDAnative kernels."""
struct FixedArray{T}
    # The number of elements in the array.
    size::Int

    # A pointer to the first element in the array.
    #
    # TODO: maybe protect this pointer from the GC somehow?
    # At the moment, this pointer is protected automatically
    # because the GC is conservative rather than precise.
    ptr::Ptr{T}
end

"""Allocates a heap-allocated array type and fills it with zeros."""
function zeros(::Type{FixedArray{T}}, size::Int) where T
    # Note: GC memory is always zero-initialized, so we don't
    # actually have to fill the array with zeros.
    bytesize = Csize_t(sizeof(T) * size)
    buf = use_gc ? gc_malloc(bytesize) : CUDAnative.malloc(bytesize)
    FixedArray{T}(size, unsafe_convert(Ptr{T}, buf))
end

"""Gets a pointer to the first element of a fixed-size array."""
function pointer(array::FixedArray{T})::Ptr{T} where T
    array.ptr
end

function getindex(array::FixedArray{T}, i::Integer)::T where T
    # TODO: bounds checking.
    unsafe_load(pointer(array), i)
end

function setindex!(array::FixedArray{T}, value::T, i::Integer) where T
    # TODO: bounds checking.
    unsafe_store!(pointer(array), value, i)
end

"""A heap-allocated matrix type, suitable for CUDAnative kernels."""
struct Matrix{Width, Height, T}
    data::FixedArray{T}
end

Matrix{Width, Height, T}() where {Width, Height, T} =
    Matrix{Width, Height, T}(zeros(FixedArray{T}, Width * Height))

function pointer(matrix::Matrix{Width, Height, T})::Ptr{T} where {Width, Height, T}
    pointer(matrix.data)
end

function getindex(matrix::Matrix{Width, Height, T}, row::Int, column::Int) where {Width, Height, T}
    getindex(matrix.data, (row - 1) * Width + column)
end

function setindex!(matrix::Matrix{Width, Height, T}, value::T, row::Int, column::Int) where {Width, Height, T}
    setindex!(matrix.data, value, (row - 1) * Width + column)
end

const matrix_dim = 50
const iterations = 20
const thread_count = 256

function kernel(result::CUDAnative.DevicePtr{Int64})
    thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    accumulator = 0

    for _ in 1:iterations
        # Allocate a matrix.
        matrix = Matrix{matrix_dim, matrix_dim, Int64}()

        # Fill it with Fibonacci numbers.
        penultimate = 0
        ultimate = 1
        for i in 1:matrix_dim
            for j in 1:matrix_dim
                matrix[i, j] = ultimate
                tmp = ultimate
                ultimate = ultimate + penultimate
                penultimate = tmp
            end
        end

        # Create a new element that contains the square of
        # every element in `matrix`.
        square = Matrix{matrix_dim, matrix_dim, Int64}()
        for i in 1:matrix_dim
            for j in 1:matrix_dim
                square[i, j] = matrix[i, j] ^ 2
            end
        end

        # Compute the sum of the squares.
        square_sum = 0
        for i in 1:matrix_dim
            for j in 1:matrix_dim
                square_sum += square[i, j]
            end
        end

        # Add that sum to an accumulator.
        accumulator += square_sum
    end

    # Write the accumulator to the result array.
    unsafe_store!(result, accumulator, thread_id)

    return
end

destination_array = Mem.alloc(Int64, thread_count)
destination_pointer = Base.unsafe_convert(CuPtr{Int64}, destination_array)

if use_gc
    time = @cuda_gc threads=thread_count kernel(destination_pointer)
    println(time)
    time = @cuda_gc threads=thread_count kernel(destination_pointer)
    println(time)
else
    time = CUDAdrv.@elapsed @cuda threads=thread_count kernel(destination_pointer)
    println(time)
    time = CUDAdrv.@elapsed @cuda threads=thread_count kernel(destination_pointer)
    println(time)
end
