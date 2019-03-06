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

export @cuda_gc, gc_malloc, gc_collect

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

# A data structure that describes a single GC "arena", i.e.,
# a section of the heap that is managed by the GC. Every arena
# has its own free list and allocation list.
struct GCArenaRecord
    # The allocation lock for the arena.
    lock_state::ReaderWriterLockState

    # The head of the free list.
    free_list_head::Ptr{GCAllocationRecord}

    # The head of the allocation list.
    allocation_list_head::Ptr{GCAllocationRecord}
end

# A reference to a Julia object.
const ObjectRef = Ptr{Nothing}

# A data structure that contains global GC info. This data
# structure is designed to be immutable: it should not be changed
# once the host has set it up.
struct GCMasterRecord
    # A pointer to the global GC arena.
    global_arena::Ptr{GCArenaRecord}

    # The maximum size of a GC root buffer, i.e., the maximum number
    # of roots per thread.
    root_buffer_capacity::UInt32

    # A pointer to a buffer that describes the number of elements
    # currently in each root buffer.
    root_buffer_sizes::Ptr{UInt32}

    # A pointer to a list of buffers that can be used to store GC roots in.
    # These root buffers are partitioned into GC frames later on.
    root_buffers::Ptr{ObjectRef}
end

# Gets the global GC interrupt lock.
@inline function get_interrupt_lock()::ReaderWriterLock
    return ReaderWriterLock(@cuda_global_ptr("gc_interrupt_lock", ReaderWriterLockState))
end

# Runs a function in such a way that no collection phases will
# run as long as the function is executing. Use with care: this
# macro acquires the GC interrupt lock in reader mode, so careless
# use may cause deadlocks.
macro nocollect(func)
    quote
        local @inline function lock_callback()
            $(esc(func))
        end

        reader_locked(lock_callback, get_interrupt_lock())
    end
end

# Gets the GC master record.
@inline function get_gc_master_record()::GCMasterRecord
    return unsafe_load(@cuda_global_ptr("gc_master_record", GCMasterRecord))
end

# Gets the thread ID of the current thread.
@inline function get_thread_id()
    return threadIdx().x
end

# Gets a pointer to the first element in the root buffer for this thread.
@inline function get_root_buffer_start()::Ptr{ObjectRef}
    master_record = get_gc_master_record()
    offset = master_record.root_buffer_capacity * get_thread_id()
    return master_record.root_buffers + offset * sizeof(ObjectRef)
end

# Same as 'new_gc_frame_impl', but does not disable collections.
function new_gc_frame_impl(size::UInt32)::Ptr{ObjectRef}
    master_record = get_gc_master_record()

    # Get the current size of the root buffer.
    current_size = unsafe_load(
        master_record.root_buffer_sizes,
        get_thread_id())

    return get_root_buffer_start() + current_size * sizeof(ObjectRef)
end

"""
    new_gc_frame(size::UInt32)::Ptr{ObjectRef}

Allocates a new GC frame.
"""
function new_gc_frame(size::UInt32)::Ptr{ObjectRef}
    @nocollect new_gc_frame_impl(size)
end

"""
    push_gc_frame(size::UInt32)

Registers a GC frame with the garbage collector.
"""
function push_gc_frame(size::UInt32)
    @nocollect begin
        master_record = get_gc_master_record()

        # Get the current size of the root buffer.
        current_size = unsafe_load(
            master_record.root_buffer_sizes,
            get_thread_id())

        # Add the new size to the current root buffer size.
        unsafe_store!(
            master_record.root_buffer_sizes,
            current_size + size,
            get_thread_id())
    end
end

"""
    pop_gc_frame(size::UInt32)

Deregisters a GC frame.
"""
function pop_gc_frame(size::UInt32)
    @nocollect begin
        master_record = get_gc_master_record()

        # Get the current size of the root buffer.
        current_size = unsafe_load(
            master_record.root_buffer_sizes,
            get_thread_id())

        # Subtract the size from the current root buffer size.
        unsafe_store!(
            master_record.root_buffer_sizes,
            current_size - size,
            get_thread_id())
    end
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

# Tries to allocate a chunk of memory in a particular GC arena.
# Returns a null pointer if no sufficiently large chunk of
# memory can be found.
function gc_malloc_local(arena::Ptr{GCArenaRecord}, bytesize::Csize_t)::Ptr{UInt8}
    # Disable collections and acquire the arena's lock.
    @nocollect begin
        arena_lock = ReaderWriterLock(@get_field_pointer(arena, :lock_state))
        result_ptr = writer_locked(arena_lock) do
            # Allocate a suitable region of memory.
            free_list_ptr = @get_field_pointer(arena, :free_list_head)::Ptr{Ptr{GCAllocationRecord}}
            allocation_list_ptr = @get_field_pointer(arena, :allocation_list_head)::Ptr{Ptr{GCAllocationRecord}}
            gc_malloc_from_free_list(free_list_ptr, allocation_list_ptr, bytesize)
        end

        # If the resulting pointer is non-null, then we'll write it to a temporary GC frame.
        # Our reasoning for doing this is that doing so ensures that the allocated memory
        # won't get collected by the GC before the caller has a chance to add it to its
        # own GC frame.
        if result_ptr != Base.unsafe_convert(Ptr{UInt8}, C_NULL)
            gc_frame = new_gc_frame_impl(UInt32(1))
            unsafe_store!(gc_frame, Base.unsafe_convert(ObjectRef, result_ptr))
        end
        return result_ptr
    end
end

"""
    gc_malloc(bytesize::Csize_t)::Ptr{UInt8}

Allocates a blob of memory that is managed by the garbage collector.
This function is designed to be called by the device.
"""
function gc_malloc(bytesize::Csize_t)::Ptr{UInt8}
    master_record = get_gc_master_record()

    # Try to malloc the object without host intervention.
    ptr = gc_malloc_local(master_record.global_arena, bytesize)
    if ptr != C_NULL
        return ptr
    end

    # We're out of memory. Ask the host to step in.
    gc_collect()

    # Try to malloc again.
    ptr = gc_malloc_local(master_record.global_arena, bytesize)
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

# The initial size of the GC heap, currently 16 MiB.
const initial_gc_heap_size = 16 * (1 << 20)

# The default capacity of a root buffer, i.e., the max number of
# roots that can be stored per thread. Currently set to
# 256 roots. That's 2 KiB of roots per thread.
const default_root_buffer_capacity = 256

# Initializes GC memory and produces a master record.
function gc_init(buffer::Array{UInt8, 1}, thread_count::Integer; root_buffer_capacity::Integer = default_root_buffer_capacity)::GCMasterRecord
    gc_memory_start_ptr = pointer(buffer, 1)
    gc_memory_end_ptr = pointer(buffer, length(buffer))

    # Set up root buffers.
    sizebuf_bytesize = sizeof(Int32) * thread_count
    sizebuf_ptr = gc_memory_start_ptr
    rootbuf_bytesize = sizeof(ObjectRef) * default_root_buffer_capacity * thread_count
    rootbuf_ptr = Base.unsafe_convert(Ptr{ObjectRef}, sizebuf_ptr + sizebuf_bytesize)

    # Compute a pointer to the start of the heap.
    heap_start_ptr = rootbuf_ptr + rootbuf_bytesize
    global_arena_size = Csize_t(gc_memory_end_ptr) - Csize_t(heap_start_ptr) - sizeof(GCAllocationRecord) - sizeof(GCArenaRecord)

    # Create a single free list entry.
    first_entry_ptr = Base.unsafe_convert(Ptr{GCAllocationRecord}, heap_start_ptr + sizeof(GCArenaRecord))
    unsafe_store!(
        first_entry_ptr,
        GCAllocationRecord(global_arena_size, C_NULL))

    # Set up the main GC data structure.
    global_arena = Base.unsafe_convert(Ptr{GCArenaRecord}, heap_start_ptr)
    unsafe_store!(
        global_arena,
        GCArenaRecord(0, first_entry_ptr, C_NULL))

    return GCMasterRecord(global_arena, root_buffer_capacity, sizebuf_ptr, rootbuf_ptr)
end

# Collects garbage. This function is designed to be called by
# the host, not by the device.
function gc_collect_impl(master_record::GCMasterRecord)
    println("GC collections are not implemented yet.")
end

# Examines a keyword argument list and gets either the value
# assigned to a key or a default value.
function get_kwarg_or_default(kwarg_list, key::Symbol, default)
    for kwarg in kwarg_list
        arg_key, val = kwarg.args
        if arg_key == key
            return val
        end
    end
    return default
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
    stream = get_kwarg_or_default(call_kwargs, :stream, CuDefaultStream())

    # Get the total number of threads.
    thread_count = get_kwarg_or_default(call_kwargs, :threads, 1)

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(code.args,
        quote
            GC.@preserve $(vars...) begin
                # Define a trivial buffer that contains the interrupt state.
                local host_interrupt_array, device_interrupt_buffer = alloc_shared_array((1,), ready)

                # Allocate a shared buffer for GC memory.
                local gc_memory_size = initial_gc_heap_size + sizeof(ObjectRef) * default_root_buffer_capacity * $(esc(thread_count))
                local host_gc_array, device_gc_buffer = alloc_shared_array((gc_memory_size,), UInt8(0))
                local master_record = gc_init(host_gc_array, $(esc(thread_count)))

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

                    # Set the GC master record.
                    try
                        global_handle = CuGlobal{GCMasterRecord}(kernel.mod, "gc_master_record")
                        set(global_handle, master_record)
                    catch exception
                        # The GC info pointer may not have been declared (because it is unused).
                        # In that case, we should do nothing.
                        if !isa(exception, CUDAdrv.CuError) || exception.code != CUDAdrv.ERROR_NOT_FOUND.code
                            rethrow()
                        end
                    end
                end

                local function handle_interrupt()
                    gc_collect_impl(master_record)
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
