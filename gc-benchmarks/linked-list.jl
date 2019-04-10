module LinkedList

using CUDAnative, CUDAdrv
import Base: foldl, reduce, sum, max, map, reverse, filter

# This benchmark constructs a linked list in a GPU kernel.
# In doing so, it stresses the allocator's ability to quickly
# allocate many small objects, as is common in idiomatic
# object-oriented programs.
# Thread divergence should be minimal in this benchmark.

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

function map_reverse(f::Function, list::List{T})::List{T} where T
    foldl(list; init=Nil{T}()) do accumulator, value
        Cons{T}(f(value), accumulator)
    end
end

function reverse(list::List{T})::List{T} where T
    map_reverse(x -> x, list)
end

function map(f::Function, list::List{T})::List{T} where T
    reverse(map_reverse(f, list))
end

function max(evaluate::Function, list::List{T}, default_value::T)::T where T
    foldl(list; init=default_value) do max_elem, elem
        if evaluate(max_elem) < evaluate(elem)
            elem
        else
            max_elem
        end
    end
end

function filter_reverse(f::Function, list::List{T})::List{T} where T
    foldl(list; init=Nil{T}()) do accumulator, value
        if f(value)
            Cons{T}(value, accumulator)
        else
            accumulator
        end
    end
end

function filter(f::Function, list::List{T})::List{T} where T
    reverse(filter_reverse(f, list))
end

const element_count = 1000
const thread_count = 32

function kernel(elements::CUDAnative.DevicePtr{Int64}, results::CUDAnative.DevicePtr{Int64})
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    l = List{Int64}(elements, element_count)
    unsafe_store!(results, sum(l), i)
    return
end

end

function linkedlist_benchmark()
    # Allocate two arrays.
    source_array = Mem.alloc(Int64, LinkedList.element_count)
    destination_array = Mem.alloc(Int64, LinkedList.thread_count)
    source_pointer = Base.unsafe_convert(CuPtr{Int64}, source_array)
    destination_pointer = Base.unsafe_convert(CuPtr{Int64}, destination_array)

    # Fill the source and destination arrays.
    Mem.upload!(source_array, Array(1:LinkedList.element_count))
    Mem.upload!(destination_array, zeros(Int64, LinkedList.thread_count))

    # Run the kernel.
    @cuda_sync threads=LinkedList.thread_count LinkedList.kernel(source_pointer, destination_pointer)

    # Verify the kernel's output.
    @test Mem.download(Int64, destination_array, LinkedList.thread_count) == repeat([sum(1:LinkedList.element_count)], LinkedList.thread_count)
end

@cuda_benchmark "linked list" linkedlist_benchmark()
