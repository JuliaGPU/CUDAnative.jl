using CUDAdrv, CUDAnative
using Random, Test
import Base: haskey, insert!

# This example defines a kernel that constructs a binary search
# tree for a set of numbers and then proceeds to test membership
# in that tree for a sequence of other numbers.
#
# The main point of this example is to demonstrate that even
# naive, pointer-chasing programs can be compiled to GPU kernels.

const use_gc = true

"""A binary search tree node."""
abstract type BinarySearchTreeNode{T} end

"""An internal node of a binary search tree."""
mutable struct InternalNode{T} <: BinarySearchTreeNode{T}
    value::T
    left::BinarySearchTreeNode{T}
    right::BinarySearchTreeNode{T}
end

InternalNode{T}(value::T) where T = InternalNode{T}(value, LeafNode{T}(), LeafNode{T}())

"""A leaf node of a binary search tree."""
mutable struct LeafNode{T} <: BinarySearchTreeNode{T} end

"""A binary search tree data structure."""
mutable struct BinarySearchTree{T}
    root::BinarySearchTreeNode{T}
end

"""Creates an empty binary search tree."""
BinarySearchTree{T}() where T = BinarySearchTree{T}(LeafNode{T}())

"""Tells if a binary search tree contains a particular element."""
function haskey(tree::BinarySearchTree{T}, value::T)::Bool where T
    walk = tree.root
    while isa(walk, InternalNode{T})
        if walk.value == value
            return true
        elseif walk.value > value
            walk = walk.right
        else
            walk = walk.left
        end
    end
    return false
end

"""Inserts an element into a binary search tree."""
function insert!(tree::BinarySearchTree{T}, value::T) where T
    if !isa(tree.root, InternalNode{T})
        tree.root = InternalNode{T}(value)
        return
    end

    walk = tree.root::InternalNode{T}
    while true
        if walk.value == value
            return
        elseif walk.value > value
            right = walk.right
            if isa(right, InternalNode{T})
                walk = right
            else
                walk.right = InternalNode{T}(value)
                return
            end
        else
            left = walk.left
            if isa(left, InternalNode{T})
                walk = left
            else
                walk.left = InternalNode{T}(value)
                return
            end
        end
    end
end

"""
Creates a binary search tree that contains elements copied from a device array.
"""
function BinarySearchTree{T}(elements::CUDAnative.DevicePtr{T}, size::Integer) where T
    tree = BinarySearchTree{T}()
    for i in 1:size
        insert!(tree, unsafe_load(elements, i))
    end
    tree
end

"""
Creates a binary search tree that contains elements copied from an array.
"""
function BinarySearchTree{T}(elements::Array{T}) where T
    tree = BinarySearchTree{T}()
    for i in 1:length(elements)
        insert!(tree, elements[i])
    end
    tree
end

# Gets a sequence of Fibonacci numbers.
function fibonacci(::Type{T}, count::Integer)::Array{T} where T
    if count == 0
        return []
    elseif count == 1
        return [one(T)]
    end

    results = [one(T), one(T)]
    for i in 1:(count - 2)
        push!(results, results[length(results) - 1] + results[length(results)])
    end
    return results
end

const number_count = 200
const thread_count = 64
const tests_per_thread = 2000

# Define a kernel that copies values using a temporary buffer.
function kernel(a::CUDAnative.DevicePtr{Int64}, b::CUDAnative.DevicePtr{Int64})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    tree = BinarySearchTree{Int64}(a, number_count)

    for j in 1:tests_per_thread
        offset = (i - 1) * tests_per_thread
        index = offset + j
        unsafe_store!(b, haskey(tree, unsafe_load(b, index)), index)
    end

    return
end

# Generate a sequence of 64-bit truncated Fibonacci numbers.
number_set = fibonacci(Int64, number_count)
# Randomize the sequence's order.
shuffle!(number_set)

# Generate numbers for which we will test membership in the sequence.
test_sequence = Array(1:(thread_count * tests_per_thread))

# Allocate two arrays.
source_array = Mem.alloc(Int64, length(number_set))
destination_array = Mem.alloc(Int64, length(test_sequence))
source_pointer = Base.unsafe_convert(CuPtr{Int64}, source_array)
destination_pointer = Base.unsafe_convert(CuPtr{Int64}, destination_array)

# Fill the source and destination arrays.
Mem.upload!(source_array, number_set)
Mem.upload!(destination_array, test_sequence)

if use_gc
    # Run the kernel.
    @cuda_gc threads=thread_count kernel(source_pointer, destination_pointer)

    # Run it again.
    Mem.upload!(destination_array, test_sequence)
    stats = @cuda_gc threads=thread_count kernel(source_pointer, destination_pointer)
else
    # Run the kernel.
    @cuda threads=thread_count kernel(source_pointer, destination_pointer)

    # Run it again and time it this time.
    Mem.upload!(destination_array, test_sequence)
    stats = CUDAdrv.@elapsed @cuda threads=thread_count kernel(source_pointer, destination_pointer)
end
println(stats)

@test Mem.download(Int64, destination_array, length(test_sequence)) == ([Int64(x in number_set) for x in test_sequence])
