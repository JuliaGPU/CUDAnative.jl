using CUDAnative, CUDAdrv
using Test
import Base: foldl, reduce, sum

# This test constructs a linked list in a GPU kernel.

use_gc = true

abstract type List{T}
end

mutable struct Nil{T} <: List{T}
end

mutable struct Cons{T} <: List{T}
    value::T
    next::List{T}
end

Cons{T}(value::T) where T = Cons{T}(value, Nil{T}())

function List{T}(pointer, count::Integer) where T
    result = Nil{T}()
    for i in count:-1:1
        result = Cons{T}(unsafe_load(pointer, i), result)
    end
    result
end

function foldl(op, list::List{T}; init) where T
    node = list
    accumulator = init
    while isa(node, Cons{T})
        accumulator = op(accumulator, node.value)
        node = node.next
    end
    accumulator
end

function reduce(op, list::List{T}; init) where T
    foldl(op, list; init=init)
end

function sum(list::List{T}) where T
    reduce(+, list; init=zero(T))
end

const element_count = 200
const thread_count = 256

function kernel(elements::CUDAnative.DevicePtr{Int64}, results::CUDAnative.DevicePtr{Int64})
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    l = List{Int64}(elements, element_count)
    unsafe_store!(results, sum(l), i)
    return
end

# Allocate two arrays.
source_array = Mem.alloc(Int64, element_count)
destination_array = Mem.alloc(Int64, thread_count)
source_pointer = Base.unsafe_convert(CuPtr{Int64}, source_array)
destination_pointer = Base.unsafe_convert(CuPtr{Int64}, destination_array)

# Fill the source and destination arrays.
Mem.upload!(source_array, Array(1:element_count))
Mem.upload!(destination_array, zeros(Int64, thread_count))

# Run the kernel.
if use_gc
    @cuda_gc threads=thread_count kernel(source_pointer, destination_pointer)
    stats = @cuda_gc threads=thread_count kernel(source_pointer, destination_pointer)
else
    @cuda threads=thread_count kernel(source_pointer, destination_pointer)
    stats = CUDAdrv.@elapsed @cuda threads=thread_count kernel(source_pointer, destination_pointer)
end
println(stats)

@test Mem.download(Int64, destination_array, thread_count) == repeat([sum(1:element_count)], thread_count)
