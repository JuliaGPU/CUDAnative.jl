using CUDAdrv, CUDAnative
using Test

# Define a kernel that copies values using a temporary buffer.
function kernel(a::CUDAnative.DevicePtr{Float32}, b::CUDAnative.DevicePtr{Float32})
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    buffer = Base.unsafe_convert(Ptr{Float32}, gc_malloc(sizeof(Float32) * Csize_t(16)))

    unsafe_store!(buffer, unsafe_load(a, i), i % 13)
    unsafe_store!(b, unsafe_load(buffer, i % 13), i)

    return
end

thread_count = 64

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
