# This file contains a GC implementation for CUDAnative kernels.
#
# CURRENT STATE OF THE GC
#
# Simple memory allocation is underway. Memory allocation currently
# uses a simple free-list.
#
# MEMORY ALLOCATION
#
# The GC's allocator uses free lists, i.e., the allocator maintains
# a list of all blocks that have not been allocated. Additionally,
# the allocator also maintains a list of all allocated blocks, so
# the collector knows which blocks it can free.
#
# END GOAL
#
# The CUDAnative GC is a precise, non-moving, mark-and-sweep GC that runs
# on the host. The device may trigger the GC via an interrupt.
#
# Some GPU-related GC implementation details:
#
#   * GC memory is shared by the host and device.
#   * Every thread gets a fixed region of memory for storing GC roots in.
#   * When the device runs out of GC memory, it requests an interrupt
#     to mark and sweep.

export @cuda_gc, gc_malloc

# A data structure that precedes every chunk of memory that has been
# allocated or put into the free list.
struct GCAllocationRecord
    # The size of the memory region this allocation record precedes.
    # This size does not include the allocation record itself.
    size::Csize_t

    # A pointer to the next allocation record in the list. If this
    # allocation record is part of the free list, then this pointer
    # points to the next free list entry; otherwise, it points to the
    # next entry in the list of allocated blocks.
    next::Ptr{GCAllocationRecord}
end

@generated function get_field_pointer_impl(base_pointer::Ptr{TBase}, ::Val{field_name}) where {TBase, field_name}
    index = Base.fieldindex(TBase, field_name)
    offset = Base.fieldoffset(TBase, index)
    type = Core.fieldtype(TBase, index)
    :(Base.unsafe_convert(Ptr{$type}, base_pointer + $(offset)))
end

# Gets a pointer to a particular field.
macro get_field_pointer(base_pointer, field_name)
    :(get_field_pointer_impl($(esc(base_pointer)), Val($field_name)))
end

# A data structure that contains information relevant
# to the GC's inner workings.
struct GCMemoryInfo
    # The head of the free list.
    free_list_head::Ptr{GCAllocationRecord}

    # The head of the allocation list.
    allocation_list_head::Ptr{GCAllocationRecord}
end

# Gets the global GC interrupt lock.
@inline function get_interrupt_lock()::ReaderWriterLock
    return ReaderWriterLock(@cuda_global_ptr("gc_interrupt_lock", ReaderWriterLockState))
end

# Gets a pointer to the global GC info data structure pointer.
@inline function get_gc_info_pointer()::Ptr{Ptr{GCMemoryInfo}}
    return @cuda_global_ptr("gc_info_pointer", Ptr{GCMemoryInfo})
end

const gc_align = Csize_t(16)

# Aligns a pointer to an alignment boundary.
function align_to_boundary(address::Ptr{T}, alignment::Csize_t = gc_align)::Ptr{T} where T
    address_int = Base.convert(Csize_t, address)
    remainder = address_int % alignment
    if remainder == Csize_t(0)
        return address
    else
        return address + alignment - remainder
    end
end

# Tries to use a free-list entry to allocate a chunk of data of size `bytesize`.
# Updates the free list if the allocation succeeds. Returns a null pointer otherwise.
function gc_use_free_list_entry(
    entry_ptr::Ptr{Ptr{GCAllocationRecord}},
    allocation_list_ptr::Ptr{Ptr{GCAllocationRecord}},
    entry::Ptr{GCAllocationRecord},
    bytesize::Csize_t,)::Ptr{UInt8}

    entry_data = unsafe_load(entry)
    if entry_data.size < bytesize
        # The entry is just too small. Return a `null` pointer.
        return C_NULL
    end

    # The entry's big enough, so we'll use it. If at all possible, we want
    # to create a new entry from any unused memory in the entry.

    # Compute the address to return.
    data_address = Base.unsafe_convert(Ptr{UInt8}, entry) + sizeof(GCAllocationRecord)

    # Compute the end of the free memory chunk.
    end_address = data_address + entry_data.size

    # Compute the start address of the new free list entry. The data
    # prefixed by the block needs to be aligned to a 16-byte boundary,
    # but the block itself doesn't.
    new_data_address = align_to_boundary(data_address + bytesize)
    new_entry_address = new_data_address - sizeof(GCAllocationRecord)
    if new_entry_address < data_address + bytesize
        new_entry_address += gc_align
    end

    # If we can place a new entry just past the allocation, then we should
    # by all means do so.
    if new_entry_address + sizeof(GCAllocationRecord) < end_address
        # Create a new free list entry.
        new_entry_size = Csize_t(end_address) - Csize_t(new_data_address)
        new_entry_ptr = Base.unsafe_convert(Ptr{GCAllocationRecord}, new_entry_address)
        unsafe_store!(
            new_entry_ptr,
            GCAllocationRecord(new_entry_size, entry_data.next))

        # Update this entry's `size` field to reflect the new entry's space
        # requirements.
        unsafe_store!(
            @get_field_pointer(entry, :size)::Ptr{Csize_t},
            Csize_t(new_entry_address) - Csize_t(data_address))

        # Update the free list pointer.
        unsafe_store!(entry_ptr, new_entry_ptr)
    else
        # We can't create a new entry, but we still have to update the free
        # list pointer.
        unsafe_store!(entry_ptr, entry_data.next)
    end

    # At this point, all we need to do is update the allocation record to
    # reflect the fact that it now represents an allocated block instead of
    # a free block.

    # Set the `next` pointer to the value stored at the allocation list pointer.
    unsafe_store!(
        @get_field_pointer(entry, :next)::Ptr{Ptr{GCAllocationRecord}},
        unsafe_load(allocation_list_ptr))

    # Update the allocation list pointer to point to the entry.
    unsafe_store!(allocation_list_ptr, entry)

    return data_address
end

# Tries to allocate a chunk of memory from a free list.
# Returns a null pointer if no sufficiently large chunk of
# memory can be found.
#
# `free_list_ptr` is a pointer to the head of the free list.
# `allocation_list_ptr` is a pointer to the head of the allocation list.
#
# This function is not thread-safe.
function gc_malloc_from_free_list(
    free_list_ptr::Ptr{Ptr{GCAllocationRecord}},
    allocation_list_ptr::Ptr{Ptr{GCAllocationRecord}},
    bytesize::Csize_t)::Ptr{UInt8}
    # To allocate memory, we will walk the free list until we find a suitable candidate.
    while free_list_ptr != C_NULL
        free_list_item = unsafe_load(free_list_ptr)

        if free_list_item == C_NULL
            break
        end

        result = gc_use_free_list_entry(free_list_ptr, allocation_list_ptr, free_list_item, bytesize)
        if result != C_NULL
            return result
        end

        free_list_ptr = @get_field_pointer(free_list_item, :next)::Ptr{Ptr{GCAllocationRecord}}
    end
    return C_NULL
end

# Tries to allocate a chunk of memory.
# Returns a null pointer if no sufficiently large chunk of
# memory can be found.
function gc_malloc_local(gc_info::Ptr{GCMemoryInfo}, bytesize::Csize_t)::Ptr{UInt8}
    # TODO: reader-lock on the interrupt lock and writer-lock on the GC's
    # lock.
    writer_locked(get_interrupt_lock()) do
        free_list_ptr = @get_field_pointer(gc_info, :free_list_head)::Ptr{Ptr{GCAllocationRecord}}
        allocation_list_ptr = @get_field_pointer(gc_info, :allocation_list_head)::Ptr{Ptr{GCAllocationRecord}}
        return gc_malloc_from_free_list(free_list_ptr, allocation_list_ptr, bytesize)
    end
end

"""
    gc_malloc(bytesize::Csize_t)::Ptr{UInt8}

Allocates a blob of memory that is managed by the garbage collector.
This function is designed to be called by the device.
"""
function gc_malloc(bytesize::Csize_t)::Ptr{UInt8}
    gc_info = unsafe_load(get_gc_info_pointer())

    # Try to malloc the object without host intervention.
    ptr = gc_malloc_local(gc_info, bytesize)
    if ptr != C_NULL
        return ptr
    end

    # We're out of memory. Ask the host to step in.
    gc_collect()

    # Try to malloc again.
    ptr = gc_malloc_local(gc_info, bytesize)
    if ptr != C_NULL
        return ptr
    end

    # Alright, so that was a spectacular failure. Let's just throw an exception.
    @cuprintf("ERROR: Out of dynamic GPU memory (trying to allocate %i bytes)\n", bytesize)
    # throw(OutOfMemoryError())
    return C_NULL
end

"""
    gc_collect()

Triggers a garbage collection phase. This function is designed
to be called by the device rather than by the host.
"""
function gc_collect()
    writer_locked(get_interrupt_lock()) do
        interrupt_or_wait()
        threadfence_system()
    end
end

# Set the initial size of the chunk of memory allocated to the
# GC to 16MiB.
const initial_gc_memory_size = 16 * (1 << 20)

# Initializes GC memory.
function gc_init(buffer::Array{UInt8, 1})
    buffer_ptr = pointer(buffer, 1)

    # Create a single free list entry.
    first_entry_ptr = Base.unsafe_convert(Ptr{GCAllocationRecord}, buffer_ptr + sizeof(GCMemoryInfo))
    unsafe_store!(
        first_entry_ptr,
        GCAllocationRecord(
            length(buffer) - sizeof(GCAllocationRecord) - sizeof(GCMemoryInfo),
            C_NULL))

    # Set up the main GC data structure.
    gc_info = Base.unsafe_convert(Ptr{GCMemoryInfo}, buffer_ptr)
    unsafe_store!(
        gc_info,
        GCMemoryInfo(first_entry_ptr, C_NULL))
end

# Collects garbage. This function is designed to be called by
# the host, not by the device.
function gc_collect_impl(info::Ptr{GCMemoryInfo})
    println("GC collections are not implemented yet.")
end

"""
    @cuda_gc [kwargs...] func(args...)

High-level interface for executing code on a GPU with GC support.
The `@cuda_gc` macro should prefix a call, with `func` a callable function
or object that should return nothing. It will be compiled to a CUDA function upon first
use, and to a certain extent arguments will be converted and anaged automatically using
`cudaconvert`. Finally, a call to `CUDAdrv.cudacall` is performed, scheduling a kernel
launch on the current CUDA context.

Several keyword arguments are supported that influence kernel compilation and execution. For
more information, refer to the documentation of respectively [`cufunction`](@ref) and
[`CUDAnative.Kernel`](@ref).
"""
macro cuda_gc(ex...)
    # destructure the `@cuda_gc` expression
    if length(ex) > 0 && ex[1].head == :tuple
        error("The tuple argument to @cuda has been replaced by keywords: `@cuda_gc threads=... fun(args...)`")
    end
    call = ex[end]
    kwargs = ex[1:end-1]

    # destructure the kernel call
    if call.head != :call
        throw(ArgumentError("second argument to @cuda_gc should be a function call"))
    end
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    compiler_kwargs, call_kwargs, env_kwargs = CUDAnative.split_kwargs(kwargs)
    vars, var_exprs = CUDAnative.assign_args!(code, args)

    # Find the stream on which the kernel is to be scheduled.
    stream = CuDefaultStream()
    for kwarg in call_kwargs
        key, val = kwarg.args
        if key == :stream
            stream = val
        end
    end

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(code.args,
        quote
            GC.@preserve $(vars...) begin
                # Define a trivial buffer that contains the interrupt state.
                local host_interrupt_array, device_interrupt_buffer = alloc_shared_array((1,), ready)

                # Allocate a shared buffer for GC memory.
                local host_gc_array, device_gc_buffer = alloc_shared_array((initial_gc_memory_size,), UInt8(0))
                gc_init(host_gc_array)

                # Define a kernel initialization function.
                local function kernel_init(kernel)
                    # Set the interrupt state pointer.
                    try
                        global_handle = CuGlobal{CuPtr{UInt32}}(kernel.mod, "interrupt_pointer")
                        set(global_handle, CuPtr{UInt32}(device_interrupt_buffer.ptr))
                    catch exception
                        # The interrupt pointer may not have been declared (because it is unused).
                        # In that case, we should do nothing.
                        if !isa(exception, CUDAdrv.CuError) || exception.code != CUDAdrv.ERROR_NOT_FOUND.code
                            rethrow()
                        end
                    end

                    # Set the GC state pointer.
                    try
                        global_handle = CuGlobal{CuPtr{GCMemoryInfo}}(kernel.mod, "gc_info_pointer")
                        set(global_handle, CuPtr{GCMemoryInfo}(device_gc_buffer.ptr))
                    catch exception
                        # The GC info pointer may not have been declared (because it is unused).
                        # In that case, we should do nothing.
                        if !isa(exception, CUDAdrv.CuError) || exception.code != CUDAdrv.ERROR_NOT_FOUND.code
                            rethrow()
                        end
                    end
                end

                local function handle_interrupt()
                    gc_collect_impl(Ptr{GCMemoryInfo}(pointer(host_gc_array, 1)))
                end

                try
                    # Standard kernel setup logic.
                    local kernel_args = CUDAnative.cudaconvert.(($(var_exprs...),))
                    local kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
                    local kernel = CUDAnative.cufunction($(esc(f)), kernel_tt; $(map(esc, compiler_kwargs)...))
                    CUDAnative.prepare_kernel(kernel; init=kernel_init, $(map(esc, env_kwargs)...))
                    kernel(kernel_args...; $(map(esc, call_kwargs)...))

                    # Handle interrupts.
                    handle_interrupts(handle_interrupt, pointer(host_interrupt_array, 1), $(esc(stream)))
                finally
                    free_shared_array(device_interrupt_buffer)
                    free_shared_array(device_gc_buffer)
                end
            end
         end)
    return code
end
