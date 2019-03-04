using CUDAdrv, CUDAnative
using Test

# Define a kernel that copies some data from one array to another.
# The host is invoked to populate the source array.
function kernel(a::CUDAnative.DevicePtr{Float32}, b::CUDAnative.DevicePtr{Float32})
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    interrupt_or_wait()
    threadfence_system()
    Base.unsafe_store!(b, Base.unsafe_load(a, i), i)
    return
end

thread_count = 64

# Allocate two arrays.
source_array = Mem.alloc(Float32, thread_count)
destination_array = Mem.alloc(Float32, thread_count)
source_pointer = Base.unsafe_convert(CuPtr{Float32}, source_array)
destination_pointer = Base.unsafe_convert(CuPtr{Float32}, destination_array)

# Zero-fill the source and destination arrays.
Mem.upload!(source_array, zeros(Float32, thread_count))
Mem.upload!(destination_array, zeros(Float32, thread_count))

# Define one stream for kernel execution and another for
# data transfer.
data_stream = CuStream()
exec_stream = CuStream()

# Define a magic value.
magic = 42.f0

# Configure the interrupt to fill the input array with the magic value.
function handle_interrupt()
    Mem.upload!(source_array, fill(magic, thread_count), data_stream; async = true)
    synchronize(data_stream)
end

# Run the kernel.
@cuda_interruptible handle_interrupt threads=thread_count stream=exec_stream kernel(source_pointer, destination_pointer)

# Check that the destination buffer is as expected.
@test Mem.download(Float32, destination_array, thread_count) == fill(magic, thread_count)
