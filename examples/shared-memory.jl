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

# Allocate a shared array.
dims = (2,4)
host_array, device_array = alloc_shared_array(dims, Int32(42))

# Write some values to the array.
host_array[1, 2] = 10
host_array[2, 1] = 0

# Check that the host's version of the array is the same as the device's.
@test host_array == Array(device_array)
