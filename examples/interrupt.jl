using CUDAdrv, CUDAnative
using Test

# Define a kernel that makes the host count.
function kernel()
    interrupt()
    return
end

thread_count = 64

# Configure the interrupt to increment a counter.
global counter = 0
function handle_interrupt()
    global counter
    counter += 1
end

# Run the kernel.
@cuda_interruptible handle_interrupt threads=thread_count kernel()

# Check that the destination buffer is as expected.
@test counter == thread_count
