using CUDAdrv, CUDAnative, CuArrays
import CUDAdrv: @apicall
using Test

# Allocates an array of host memory that is page-locked and accessible
# to the device. Maps the allocation into the CUDA address space.
# Returns a (host array, CuArray) pair. The former can be used by
# the host to access the array, the latter can be used by the device.
function alloc_shared_array(dims::Tuple{Vararg{Int64, N}}, init::T) where {T, N}
    # Allocate memory that is accessible to both the host and the device.
    bytesize = prod(dims) * sizeof(T)
    ptr_ref = Ref{Ptr{Cvoid}}()
    @apicall(
        :cuMemAllocHost,
        (Ptr{Ptr{Cvoid}}, Csize_t),
        ptr_ref, bytesize)
    device_buffer = CUDAdrv.Mem.Buffer(ptr_ref[], bytesize, CuCurrentContext())

    # Wrap the memory in an array for the host.
    host_array = Base.unsafe_wrap(Array{T, N}, Ptr{T}(ptr_ref[]), dims; own = false)

    # Initialize the array's contents.
    fill!(host_array, init)

    return host_array, CuArray{T, N}(device_buffer, dims; own = false)
end

# This example shows that devices can communicate with the host
# and vice-versa *during* the execution of a kernel.
#
# What happens is, in chronological order:
#
#   1. A buffer is zero-initialized by the host.
#   2. A kernel is started on the device; said kernel
#      waits for the buffer to become nonzero.
#   3. The host makes the buffer nonzero.
#   4. The kernel exists once the buffer is nonzero.
#

function spin(a)
    i = threadIdx().x + blockDim().x * (blockIdx().x-1)
    # Make sure that 'a[i]' is actually zero when we get started.
    if a[i] != 0.f0
        return
    end

    # We wait for the host to set 'a[i]' to a nonzero value.
    while true
        ccall("llvm.nvvm.membar.gl", llvmcall, Cvoid, ())
        if a[i] != 0.f0
            break
        end
    end
    # Next, we set 'a[i]' to some magic value.
    a[i] = 42.f0
    return
end

# Allocate a shared array.
dims = (3,4)
host_array, device_array = alloc_shared_array(dims, 0.f0)

# Launch the kernel.
@cuda threads=prod(dims) spin(device_array)

# Go to sleep for a few milliseconds, to make sure
# that the kernel will have started already.
sleep(0.2)

# Fill the array with ones now to unblock the kernel.
fill!(host_array, 1.f0)

# Wait for the kernel to exit.
synchronize()

# Check that the array has been set to the magic value.
@test host_array == fill(42.f0, dims)
