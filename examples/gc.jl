using CUDAdrv, CUDAnative
using Test

mutable struct TempStruct
    data::Float32
end

@noinline function escape(val)
    Base.pointer_from_objref(val)
end

function upload!(destination, source)
    Mem.copy!(destination, pointer(source), sizeof(source))
end

function download(::Type{T}, source, dims) where T
    result = Array{T}(undef, dims)
    Mem.copy!(pointer(result), source, sizeof(result))
    result
end

# Define a kernel that copies values using a temporary struct.
function kernel(a::CUDAnative.DevicePtr{Float32}, b::CUDAnative.DevicePtr{Float32})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    for j in 1:2
        # Allocate a mutable struct and make sure it ends up on the GC heap.
        temp = TempStruct(unsafe_load(a, i))
        escape(temp)

        # Allocate a large garbage buffer to force collections.
        gc_malloc(Csize_t(256 * 1024))

        # Use the mutable struct. If its memory has been reclaimed (by accident)
        # then we expect the test at the end of this file to fail.
        unsafe_store!(b, temp.data, i)
    end

    return
end

thread_count = 256

# Allocate two arrays.
source_array = Mem.alloc(Mem.DeviceBuffer, sizeof(Float32) * thread_count)
destination_array = Mem.alloc(Mem.DeviceBuffer, sizeof(Float32) * thread_count)
source_pointer = Base.unsafe_convert(CuPtr{Float32}, source_array)
destination_pointer = Base.unsafe_convert(CuPtr{Float32}, destination_array)

# Fill the source and destination arrays.
upload!(source_array, fill(42.f0, thread_count))
upload!(destination_array, zeros(Float32, thread_count))

# Run the kernel.
@cuda gc=true threads=thread_count kernel(source_pointer, destination_pointer)

@test download(Float32, destination_array, thread_count) == fill(42.f0, thread_count)
