using CUDAdrv, CUDAnative
using Test

mutable struct TempStruct
    data::Float32
end

@noinline function escape(val)
    Base.pointer_from_objref(val)
end

# Define a kernel that copies values using a temporary buffer.
function kernel(a::CUDAnative.DevicePtr{Float32}, b::CUDAnative.DevicePtr{Float32})
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    for j in 1:256
        # Allocate a mutable struct and make sure it ends up on the GC heap.
        temp = TempStruct(unsafe_load(a, i))
        escape(temp)
        unsafe_store!(b, temp.data, i)
    end

    return
end

thread_count = 256

# Allocate two arrays.
source_array = Mem.alloc(Float32, thread_count)
destination_array = Mem.alloc(Float32, thread_count)
source_pointer = Base.unsafe_convert(CuPtr{Float32}, source_array)
destination_pointer = Base.unsafe_convert(CuPtr{Float32}, destination_array)

# Fill the source and destination arrays.
Mem.upload!(source_array, fill(42.f0, thread_count))
Mem.upload!(destination_array, zeros(Float32, thread_count))

# Run the kernel.
@cuda_gc threads=thread_count kernel(source_pointer, destination_pointer)

@test Mem.download(Float32, destination_array, thread_count) == fill(42.f0, thread_count)
