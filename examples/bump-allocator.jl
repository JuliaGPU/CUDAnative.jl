using Test

using CUDAdrv, CUDAnative
include(joinpath(@__DIR__, "..", "test", "array.jl"))   # real applications: use CuArrays.jl

mutable struct Box{T}
    value::T
end

@noinline function escape(obj)
    Base.pointer_from_objref(obj)
end

function vcopy(a, b)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    box = Box(a[i])
    escape(box)
    b[i] = box.value
    return
end

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = similar(a)

d_a = CuTestArray(a)
d_b = CuTestArray(b)

len = prod(dims)

# Allocate a 1 MiB heap for the bump allocator.
heap_capacity = 1024 * 1024
heap = Mem.alloc(Mem.DeviceBuffer, heap_capacity)
heap_start_address = pointer(heap)
# Create an initialization callback for the bump allocator.
function init(kernel)
    CUDAnative.Runtime.bump_alloc_init!(kernel, heap_start_address, heap_capacity)
end
# Run the kernel.
@cuda threads=len init=init malloc="ptx_bump_alloc" vcopy(d_a, d_b)
# Free the heap.
Mem.free(heap)

b = Array(d_b)
@test a ≈ b
