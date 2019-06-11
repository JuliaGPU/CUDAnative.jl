# This file contains a GC implementation for CUDAnative kernels.
# The sections below contain some basic info on how the garbage
# collector works.
#
# MEMORY ALLOCATION
#
# The GC's allocator uses free lists, i.e., the allocator maintains
# a list of all blocks that have not been allocated. Additionally,
# the allocator also maintains a list of all allocated blocks, so
# the collector knows which blocks it can free.
#
# GARBAGE COLLECTION
#
# The garbage collector itself is a semi-conservative, non-moving,
# mark-and-sweep, stop-the-world GC that runs on the host.
# The device may trigger the GC via an interrupt.
#
# The GC is semi-conservative in the sense that its set of roots
# is precise but objects are scanned in an imprecise way.
#
# After every garbage collection, the GC will compact free lists:
# adjacent free list block will be merged and the free list will
# be sorted based on block sizes to combat memory fragmentation.
#
# If a free list is deemed to be "starving" after a collection, i.e.,
# its total amount of free bytes has dropped below some threshold,
# then a fresh chunk of GC-managed memory is allocated and added to
# the free list.
#
# SAFEPOINTS
#
# Every warp gets a flag that tells if that warp is in a safepoint.
# When a collection is triggered, the collector waits for every warp
# to reach a safepoint. The warps indicate that they have reached a
# safepoint by setting the flag.
#
# MISCELLANEOUS
#
# Some miscellaneous GPU-related GC implementation details:
#
#   * GC memory is shared by the host and device.
#   * Every thread gets a fixed region of memory for storing GC roots in.
#   * When the device runs out of GC memory, it requests an interrupt
#     to mark and sweep.

export gc_malloc, gc_malloc_object, gc_collect, gc_safepoint, GCConfiguration

import Base: length, show
import Printf: @sprintf

# A data structure that precedes every chunk of memory that has been
# allocated or put into the free list.
struct FreeListRecord
    # The size of the memory region this allocation record precedes.
    # This size does not include the allocation record itself.
    size::Csize_t

    # A pointer to the next allocation record in the list. If this
    # allocation record is part of the free list, then this pointer
    # points to the next free list entry; otherwise, it points to the
    # next entry in the list of allocated blocks.
    next::Ptr{FreeListRecord}
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

# Gets a pointer to the first byte of data managed by an allocation record.
function data_pointer(record::Ptr{FreeListRecord})::Ptr{UInt8}
    Base.unsafe_convert(Ptr{UInt8}, record) + sizeof(FreeListRecord)
end

# Takes a pointer to the first byte of data managed by an allocation record
# and produces a pointer to the record itself.
function record_pointer(data::Ptr{UInt8})::Ptr{FreeListRecord}
    Base.unsafe_convert(Ptr{FreeListRecord}, record) - sizeof(FreeListRecord)
end

# Gets a pointer to the first byte of data no longer managed by an allocation record.
function data_end_pointer(record::Ptr{FreeListRecord})::Ptr{UInt8}
    data_pointer(record) + unsafe_load(@get_field_pointer(record, :size))
end

# A data structure that describes a single GC "arena", i.e.,
# a section of the heap that is managed by the GC. Every arena
# has its own free list and allocation list.
struct FreeListArena
    # The allocation lock for the arena.
    lock_state::ReaderWriterLockState

    # The head of the free list.
    free_list_head::Ptr{FreeListRecord}

    # The head of the allocation list.
    allocation_list_head::Ptr{FreeListRecord}
end

# Gets a free list arena's lock.
get_lock(arena::Ptr{FreeListArena}) = ReaderWriterLock(@get_field_pointer(arena, :lock_state))

const gc_align = Csize_t(16)

# Aligns a pointer to an alignment boundary.
function align_downward(address::Ptr{T}, alignment::Csize_t = gc_align)::Ptr{T} where T
    address_int = Base.convert(Csize_t, address)
    remainder = address_int % alignment
    if remainder == Csize_t(0)
        return address
    else
        return address + alignment - remainder
    end
end

# Aligns a pointer to an alignment boundary.
function align_upward(address::Ptr{T}, alignment::Csize_t = gc_align)::Ptr{T} where T
    result = align_downward(address, alignment)
    if result < address
        result += alignment
    end
    result
end

# Aligns a pointer to an alignment boundary.
function align_upward(offset::T, alignment::Csize_t = gc_align)::T where T <: Integer
    convert(T, Csize_t(align_upward(convert(Ptr{UInt8}, Csize_t(offset)), alignment)))
end

# Gets the size of an aligned header, including padding to satisfy
# alignment requirements.
@generated function header_size(::Type{T}, ::Val{alignment} = Val(gc_align))::UInt32 where {T, alignment}
    result = align_upward(UInt32(sizeof(T)), alignment)
    :($result)
end

# A reference to a Julia object.
const ObjectRef = Ptr{Nothing}

# A GC frame is just a pointer to an array of Julia objects.
const GCFrame = Ptr{ObjectRef}

# The states a safepoint flag can have.
@enum SafepointState::UInt32 begin
    # Indicates that a warp is not in a safepoint.
    not_in_safepoint = 0
    # Indicates that a warp is in a safepoint. This
    # flag will be reset to `not_in_safepoint` by the
    # collector on the next collecotr.
    in_safepoint = 1
    # Indicates that a warp is in a perma-safepoint:
    # the collector will not try to set this type
    # of safepoint back to `not_in_safepoint`.
    in_perma_safepoint = 2
end

const LocalArena = FreeListArena
const GlobalArena = FreeListArena

# A data structure that contains global GC info. This data
# structure is designed to be immutable: it should not be changed
# once the host has set it up.
struct GCMasterRecord
    # The number of warps.
    warp_count::UInt32

    # The number of threads.
    thread_count::UInt32

    # The maximum size of a GC root buffer, i.e., the maximum number
    # of roots per thread.
    root_buffer_capacity::UInt32

    # The number of local arenas.
    local_arena_count::UInt32

    # A pointer to a list of local GC arena pointers.
    local_arenas::Ptr{Ptr{LocalArena}}

    # A pointer to the global GC arena.
    global_arena::Ptr{GlobalArena}

    # A pointer to a list of safepoint flags. Every warp has its
    # own flag.
    safepoint_flags::Ptr{SafepointState}

    # A pointer to a list of root buffer pointers that point to the
    # end of the root buffer for every thread.
    root_buffer_fingers::Ptr{Ptr{ObjectRef}}

    # A pointer to a list of buffers that can be used to store GC roots in.
    # These root buffers are partitioned into GC frames later on.
    root_buffers::Ptr{ObjectRef}
end

# Iterates through all arena pointers stored in a GC master record.
@inline function iterate_arenas(fun::Function, master_record::GCMasterRecord)
    for i in 1:master_record.local_arena_count
        fun(unsafe_load(master_record.local_arenas, i))
    end
    fun(master_record.global_arena)
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

# Gets a pointer to the local arena for this thread. This
# pointer may be null if there are no local arenas.
@inline function get_local_arena()::Ptr{LocalArena}
    master_record = get_gc_master_record()
    if master_record.local_arena_count == UInt32(0)
        return Base.unsafe_convert(Ptr{LocalArena}, C_NULL)
    else
        return unsafe_load(
            master_record.local_arenas,
            ((get_warp_id() - 1) % master_record.local_arena_count) + 1)
    end
end

"""
    new_gc_frame(size::UInt32)::GCFrame

Allocates a new GC frame.
"""
@inline function new_gc_frame(size::UInt32)::GCFrame
    master_record = get_gc_master_record()
    # Return the root buffer tip: that's where the new GC frame starts.
    return unsafe_load(master_record.root_buffer_fingers, get_thread_id())
end

"""
    push_gc_frame(gc_frame::GCFrame, size::UInt32)

Registers a GC frame with the garbage collector.
"""
@inline function push_gc_frame(gc_frame::GCFrame, size::UInt32)
    master_record = get_gc_master_record()

    # Update the root buffer tip.
    unsafe_store!(
        master_record.root_buffer_fingers,
        gc_frame + size * sizeof(ObjectRef),
        get_thread_id())
    return
end

"""
    pop_gc_frame(gc_frame::GCFrame)

Deregisters a GC frame.
"""
@inline function pop_gc_frame(gc_frame::GCFrame)
    master_record = get_gc_master_record()

    # Update the root buffer tip.
    unsafe_store!(
        master_record.root_buffer_fingers,
        gc_frame,
        get_thread_id())
    return
end

"""
    gc_safepoint()

Signals that this warp has reached a GC safepoint.
"""
function gc_safepoint()
    wait_for_interrupt() do
        gc_set_safepoint_flag(in_safepoint; overwrite = false)
    end
    return
end

"""
    gc_perma_safepoint()

Signals that this warp has reached a GC perma-safepoint:
the GC doesn't need to wait for this warp to reach a safepoint
before starting collections. Instead, the GC may assume that
the warp is already in a safepoint.

Be careful with this function: all bets are off when this
function is used improperly. For a more controlled (but still
super dangerous) way to use perma-safepoints, see the
`@perma_safepoint` macro.
"""
function gc_perma_safepoint()
    gc_set_safepoint_flag(in_perma_safepoint)
    return
end

# Sets this warp's safepoint flag to a particular state.
function gc_set_safepoint_flag(value::SafepointState; overwrite::Bool = true)
    master_record = get_gc_master_record()
    warp_id = get_warp_id()
    safepoint_flag_ptr = master_record.safepoint_flags + sizeof(SafepointState) * (warp_id - 1)
    if overwrite
        volatile_store!(safepoint_flag_ptr, value)
    else
        atomic_compare_exchange!(safepoint_flag_ptr, not_in_safepoint, value)
    end
    return
end

# Marks a region as a perma-safepoint: the entire region
# is a safepoint. Note that perma-safepoints are not allowed
# to include non-perma-safepoints.
macro perma_safepoint(expr)
    quote
        gc_perma_safepoint()
        local result = $(esc(expr))
        gc_set_safepoint_flag(not_in_safepoint)
        result
    end
end

# Tries to use a free-list entry to allocate a chunk of data of size `bytesize`,
# producing an appropriately-sized free list entry that prefixes the data. This
# entry is removed from the free list but not yet added to the allocation list.
function gc_take_list_entry(
    entry_ptr::Ptr{Ptr{FreeListRecord}},
    entry::Ptr{FreeListRecord},
    bytesize::Csize_t)::Ptr{FreeListRecord}

    entry_data = unsafe_load(entry)
    if entry_data.size < bytesize
        # The entry is just too small. Return a `null` pointer.
        return C_NULL
    end

    # The entry's big enough, so we'll use it. If at all possible, we want
    # to create a new entry from any unused memory in the entry.

    # Compute the address to return.
    data_address = data_pointer(entry)

    # Compute the end of the free memory chunk.
    end_address = data_address + entry_data.size

    # Compute the start address of the new free list entry. The data
    # prefixed by the block needs to be aligned to a 16-byte boundary,
    # but the block itself doesn't.
    new_data_address = align_downward(data_address + bytesize)
    new_entry_address = new_data_address - sizeof(FreeListRecord)
    if new_entry_address < data_address + bytesize
        new_entry_address += gc_align
        new_data_address += gc_align
    end

    # If we can place a new entry just past the allocation, then we should
    # by all means do so.
    if new_data_address < end_address
        # Create a new free list entry.
        new_entry_size = Csize_t(end_address) - Csize_t(new_data_address)
        new_entry_ptr = Base.unsafe_convert(Ptr{FreeListRecord}, new_entry_address)
        unsafe_store!(
            new_entry_ptr,
            FreeListRecord(new_entry_size, entry_data.next))

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

    return entry
end

# Prepends a free list record to a free list.
function gc_add_to_free_list(
    entry::Ptr{FreeListRecord},
    list_ptr::Ptr{Ptr{FreeListRecord}})

    # Set the `next` pointer to the value stored at the allocation list pointer.
    unsafe_store!(
        @get_field_pointer(entry, :next)::Ptr{Ptr{FreeListRecord}},
        unsafe_load(list_ptr))

    # Update the allocation list pointer to point to the entry.
    unsafe_store!(list_ptr, entry)
end

# Tries to allocate a chunk of memory from a free list.
# Returns a null pointer if no sufficiently large chunk of
# memory can be found.
# If the result is non-null, then a free list record is
# returned that has been taken from the free list but not
# yet added to another list.
function gc_take_any_list_entry(
    free_list_ptr::Ptr{Ptr{FreeListRecord}},
    bytesize::Csize_t)::Ptr{FreeListRecord}

    # To allocate memory, we will walk the free list until we find a suitable candidate.
    while true
        free_list_item = unsafe_load(free_list_ptr)

        if free_list_item == C_NULL
            return C_NULL
        end

        result = gc_take_list_entry(free_list_ptr, free_list_item, bytesize)
        if result != C_NULL
            return result
        end

        free_list_ptr = @get_field_pointer(free_list_item, :next)::Ptr{Ptr{FreeListRecord}}
    end
end

# Tries to allocate a chunk of memory from a free list.
# Returns a null pointer if no sufficiently large chunk of
# memory can be found.
#
# This function is not thread-safe.
function gc_malloc_from_free_list(arena::Ptr{FreeListArena}, bytesize::Csize_t)::Ptr{UInt8}
    free_list_ptr = @get_field_pointer(arena, :free_list_head)::Ptr{Ptr{FreeListRecord}}
    allocation_list_ptr = @get_field_pointer(arena, :allocation_list_head)::Ptr{Ptr{FreeListRecord}}

    # Try to take the entry out of the free list.
    result_entry = gc_take_any_list_entry(free_list_ptr, bytesize)
    if result_entry == C_NULL
        # The entry is just too small. Return a `null` pointer.
        return C_NULL
    end

    # At this point, all we need to do is update the allocation record to
    # reflect the fact that it now represents an allocated block instead of
    # a free block.
    gc_add_to_free_list(result_entry, allocation_list_ptr)

    return data_pointer(result_entry)
end

# Writes a pointer to a temporary GC frame. This will keep the pointer
# from getting collected until the caller has a chance to add it to its
# own GC frame.
function gc_protect(pointer::Ptr{UInt8})
    if pointer != Base.unsafe_convert(Ptr{UInt8}, C_NULL)
        gc_frame = new_gc_frame(UInt32(1))
        unsafe_store!(gc_frame, Base.unsafe_convert(ObjectRef, pointer))
    end
end

# Tries to allocate a chunk of memory in a particular GC arena.
# Returns a null pointer if no sufficiently large chunk of
# memory can be found.
function gc_malloc_local(arena::Ptr{FreeListArena}, bytesize::Csize_t; acquire_lock=true)::Ptr{UInt8}
    # Acquire the arena's lock.
    result_ptr = writer_locked(get_lock(arena); acquire_lock=acquire_lock) do
        # Allocate a suitable region of memory.
        gc_malloc_from_free_list(arena, bytesize)
    end

    # If the resulting pointer is non-null, then we'll write it to a temporary GC frame.
    # Our reasoning for doing this is that doing so ensures that the allocated memory
    # won't get collected by the GC before the caller has a chance to add it to its
    # own GC frame.
    gc_protect(result_ptr)
    return result_ptr
end

# Transfers a block of free memory from one arena to another and then
# allocates a differently-sized block of memory from the destination
# arena.
function gc_transfer_and_malloc(
    from_arena::Ptr{FreeListArena},
    to_arena::Ptr{FreeListArena},
    transfer_bytesize::Csize_t,
    alloc_bytesize::Csize_t)::Ptr{UInt8}

    from_free_list = @get_field_pointer(from_arena, :free_list_head)::Ptr{Ptr{FreeListRecord}}
    entry = writer_locked(get_lock(from_arena)) do
        # Try to take the entry out of the free list.
        gc_take_any_list_entry(from_free_list, transfer_bytesize)
    end

    if entry == C_NULL
        return C_NULL
    else
        to_free_list = @get_field_pointer(to_arena, :free_list_head)::Ptr{Ptr{FreeListRecord}}
        return writer_locked(get_lock(to_arena)) do
            gc_add_to_free_list(entry, to_free_list)
            gc_malloc_local(to_arena, alloc_bytesize; acquire_lock=false)
        end
    end
end

"""
    gc_malloc(bytesize::Csize_t)::Ptr{UInt8}

Allocates a blob of memory that is managed by the garbage collector.
This function is designed to be called by the device.
"""
function gc_malloc(bytesize::Csize_t)::Ptr{UInt8}
    master_record = get_gc_master_record()

    function allocate()
        # Try to allocate in the local arena second. If that doesn't
        # work, we'll move on to the global arena, which is bigger but
        # is shared by all threads. (We want to minimize contention
        # on the global arena's lock.)
        local_arena = get_local_arena()
        if local_arena != C_NULL
            local_ptr = gc_malloc_local(local_arena, bytesize)
            if local_ptr != C_NULL
                return local_ptr
            end
        else
            # If there is no local arena then we will just have to allocate
            # from the global arena directly.
            return gc_malloc_local(master_record.global_arena, bytesize)
        end

        # Try to use the global arena if all else fails, but only if the chunk
        # of memory we want to allocate is sufficiently large. Allocating lots of
        # small chunks in the global arena will result in undue contention and slow
        # down kernels dramatically.
        #
        # If we need to allocate a small chunk of memory but the local arena is
        # empty, then we will transfer a *much* larger chunk of memory from the global
        # arena to the local arena. After that we'll allocate in the local arena.
        min_global_alloc_size = Csize_t(256 * (1 << 10))
        if bytesize >= min_global_alloc_size
            local_ptr = gc_malloc_local(master_record.global_arena, bytesize)
        else
            local_ptr = gc_transfer_and_malloc(
                master_record.global_arena,
                local_arena,
                min_global_alloc_size,
                bytesize)
        end
        return local_ptr
    end

    # Try to malloc the object without host intervention.
    ptr = @perma_safepoint @nocollect allocate()
    if ptr != C_NULL
        return ptr
    end

    # We're out of memory, which means that we need the garbage collector
    # to step in. Set a perma-safepoint and acquire the interrupt lock.
    ptr = @perma_safepoint writer_locked(get_interrupt_lock()) do
        # Try to allocate memory again. This is bound to fail for the
        # first thread that acquires the interrupt lock, but it is quite
        # likely to succeed if we are *not* in the first thread that
        # acquired the garbage collector lock.
        ptr2 = allocate()

        if ptr2 == C_NULL
            # We are either the first thread to acquire the interrupt lock
            # or the additional memory produced by a previous collection has
            # already been exhausted. Trigger the garbage collector.
            gc_collect_impl()

            # Try to malloc again.
            ptr2 = gc_malloc_local(master_record.global_arena, bytesize)
        end
        ptr2
    end
    if ptr != C_NULL
        return ptr
    end

    # Alright, so that was a spectacular failure. Let's just throw an exception.
    @cuprintf("ERROR: Out of GPU GC memory (trying to allocate %i bytes)\n", bytesize)
    # throw(OutOfMemoryError())
    return C_NULL
end

"""
    gc_malloc_object(bytesize::Csize_t)

Allocates an object that is managed by the garbage collector.
This function is designed to be called by the device.
"""
function gc_malloc_object(bytesize::Csize_t)
    unsafe_pointer_to_objref(gc_malloc(bytesize))
end

# Zero-fills a range of memory.
function zero_fill!(start_ptr::Ptr{UInt8}, size::Csize_t)
    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), start_ptr, 0, size)
end

# Zero-fills a range of memory.
function zero_fill!(start_ptr::Ptr{UInt8}, end_ptr::Ptr{UInt8})
    zero_fill!(start_ptr, Csize_t(end_ptr) - Csize_t(start_ptr))
end

# Tries to free a block of memory from a particular arena. `record_ptr`
# must point to a pointer to the GC allocation record to free. It will
# be updated to point to the next allocation.
#
# This function is designed to be called by the host: it does not
# turn off collections. It can be called by the device, but in that
# case it should be prefixed by the `@nocollect` macro followed by
# a write lock acquisition on the arena's lock.
function gc_free_local(
    arena::Ptr{FreeListArena},
    record_ptr::Ptr{Ptr{FreeListRecord}})

    record = unsafe_load(record_ptr)
    next_record_ptr = @get_field_pointer(record, :next)
    free_list_head_ptr = @get_field_pointer(arena, :free_list_head)

    # Remove the record from the allocation list.
    unsafe_store!(record_ptr, unsafe_load(next_record_ptr))

    # Add the record to the free list and update its `next` pointer
    # (but not in that order).
    unsafe_store!(next_record_ptr, unsafe_load(free_list_head_ptr))
    unsafe_store!(free_list_head_ptr, record)

    # Zero-fill the newly freed block of memory.
    zero_fill!(data_pointer(record), unsafe_load(@get_field_pointer(record, :size)))
end

# Like 'gc_collect', but does not acquire the interrupt lock.
function gc_collect_impl()
    interrupt_or_wait()
    threadfence_system()
end

"""
    gc_collect()

Triggers a garbage collection phase. This function is designed
to be called by the device rather than by the host.
"""
function gc_collect()
    writer_locked(gc_collect_impl, get_interrupt_lock())
end

# One megabyte.
const MiB = 1 << 20

# A description of a region of memory that has been allocated to the GC heap.
const GCHeapRegion = CUDAdrv.Mem.HostBuffer

# A description of all memory that has been allocated to the GC heap.
struct GCHeapDescription
    # A list of the set of regions that comprise the GC heap.
    regions::Array{GCHeapRegion, 1}
end

GCHeapDescription() = GCHeapDescription([])

# A data structure that contains GC configuration parameters.
struct GCConfiguration
    # The number of local arenas to create.
    local_arena_count::Int

    # The max number of roots that can be stored per thread.
    root_buffer_capacity::Int

    # The point at which the global arena is deemed to be starving, i.e.,
    # it no longer contains enough memory to perform basic allocations.
    # If the global arena's free byte count stays below the arena starvation
    # threshold after a collection phase, the collector will allocate
    # additional memory to the arena such that it is no longer starving.
    global_arena_starvation_threshold::Int

    # The initial size of the global arena, in bytes.
    global_arena_initial_size::Int

    # The point at which a local arena is deemed to be starving, i.e.,
    # it no longer contains enough memory to perform basic allocations.
    # If a local arena's free byte count stays below the arena starvation
    # threshold after a collection phase, the collector will allocate
    # additional memory to the arena such that it is no longer starving.
    local_arena_starvation_threshold::Int

    # The initial size of a local arena, in bytes.
    local_arena_initial_size::Int
end

# Creates a GC configuration.
function GCConfiguration(;
    local_arena_count::Integer = 8,
    root_buffer_capacity::Integer = 256,
    global_arena_starvation_threshold::Integer = 4 * MiB,
    global_arena_initial_size::Integer = 2 * MiB,
    local_arena_starvation_threshold::Integer = 1 * MiB,
    local_arena_initial_size::Integer = 1 * MiB)

    GCConfiguration(
        local_arena_count,
        root_buffer_capacity,
        global_arena_starvation_threshold,
        global_arena_initial_size,
        local_arena_starvation_threshold,
        local_arena_initial_size)
end

function initial_heap_size(config::GCConfiguration, thread_count::Integer)
    warp_count = Base.ceil(UInt32, thread_count / CUDAdrv.warpsize(device()))
    local_arenas_bytesize = sizeof(Ptr{LocalArena}) * config.local_arena_count
    safepoint_bytesize = sizeof(SafepointState) * warp_count
    fingerbuf_bytesize = sizeof(Ptr{ObjectRef}) * thread_count
    rootbuf_bytesize = sizeof(ObjectRef) * config.root_buffer_capacity * thread_count

    result = 0
    result += local_arenas_bytesize
    result += safepoint_bytesize
    result += fingerbuf_bytesize
    result += rootbuf_bytesize
    result += config.local_arena_count * config.local_arena_initial_size
    result += config.global_arena_initial_size
    return result
end

# Initializes a GC heap and produces a master record.
function gc_init!(
    heap::GCHeapDescription,
    config::GCConfiguration,
    thread_count::Integer)::GCMasterRecord

    warp_count = Base.ceil(UInt32, thread_count / CUDAdrv.warpsize(device()))

    master_region = heap.regions[1]

    gc_memory_start_ptr = pointer(master_region)
    gc_memory_end_ptr = pointer(master_region) + sizeof(master_region)

    # Allocate a local arena pointer buffer.
    local_arenas_bytesize = sizeof(Ptr{LocalArena}) * config.local_arena_count
    local_arenas_ptr = Base.unsafe_convert(Ptr{Ptr{LocalArena}}, gc_memory_start_ptr)

    # Allocate the safepoint flag buffer.
    safepoint_bytesize = sizeof(SafepointState) * warp_count
    safepoint_ptr = Base.unsafe_convert(Ptr{SafepointState}, local_arenas_ptr + local_arenas_bytesize)

    # Allocate root buffers.
    fingerbuf_bytesize = sizeof(Ptr{ObjectRef}) * thread_count
    fingerbuf_ptr = Base.unsafe_convert(Ptr{Ptr{ObjectRef}}, safepoint_ptr + fingerbuf_bytesize)
    rootbuf_bytesize = sizeof(ObjectRef) * config.root_buffer_capacity * thread_count
    rootbuf_ptr = Base.unsafe_convert(Ptr{ObjectRef}, fingerbuf_ptr + fingerbuf_bytesize)

    # Populate the root buffer fingers.
    for i in 1:thread_count
        unsafe_store!(fingerbuf_ptr, rootbuf_ptr + (i - 1) * sizeof(ObjectRef) * config.root_buffer_capacity, i)
    end

    # Compute a pointer to the start of the tiny arena.
    arena_start_ptr = rootbuf_ptr + rootbuf_bytesize

    # Set up local arenas.
    for i in 1:config.local_arena_count
        local_arena = make_gc_arena!(LocalArena, arena_start_ptr, Csize_t(config.local_arena_initial_size))
        unsafe_store!(local_arenas_ptr, local_arena, i)
        arena_start_ptr += config.local_arena_initial_size
    end

    # Set up the global arena.
    global_arena = make_gc_arena!(GlobalArena, arena_start_ptr, Csize_t(gc_memory_end_ptr) - Csize_t(arena_start_ptr))

    return GCMasterRecord(
        warp_count,
        UInt32(thread_count),
        UInt32(config.root_buffer_capacity),
        UInt32(config.local_arena_count),
        local_arenas_ptr,
        global_arena,
        safepoint_ptr,
        fingerbuf_ptr,
        rootbuf_ptr)
end

# Takes a zero-filled region of memory and turns it into a block
# managed by the GC, prefixed with an allocation record.
function make_gc_block!(start_ptr::Ptr{T}, size::Csize_t)::Ptr{FreeListRecord} where T
    entry = Base.unsafe_convert(Ptr{FreeListRecord}, start_ptr)
    unsafe_store!(
        entry,
        FreeListRecord(
            Csize_t(start_ptr + size) - Csize_t(data_pointer(entry)),
            C_NULL))
    return entry
end

# Takes a zero-filled region of memory and turns it into an arena
# managed by the GC, prefixed with an arena record.
function make_gc_arena!(::Type{FreeListArena}, start_ptr::Ptr{T}, size::Csize_t)::Ptr{FreeListArena} where T
    # Create a single free list entry.
    first_entry_ptr = make_gc_block!(start_ptr + sizeof(FreeListArena), size - sizeof(FreeListArena))

    # Set up the arena record.
    arena = Base.unsafe_convert(Ptr{FreeListArena}, start_ptr)
    unsafe_store!(
        arena,
        FreeListArena(0, first_entry_ptr, C_NULL))

    arena
end

# Tells if a GC heap contains a particular pointer.
function contains(heap::GCHeapDescription, pointer::Ptr{T}) where T
    for region in heap.regions
        if pointer >= pointer(region) && pointer < pointer(region) + sizeof(region)
            return true
        end
    end
    return false
end

# Expands the GC heap by allocating a region of memory and adding it to
# the list of allocated regions. `size` describes the amount of bytes to
# allocate. Returns the allocated region.
function expand!(heap::GCHeapDescription, size::Integer)::GCHeapRegion
    region = CUDAdrv.Mem.alloc(CUDAdrv.Mem.HostBuffer, size, CUDAdrv.Mem.HOSTALLOC_DEVICEMAP)
    push!(heap.regions, region)
    return region
end

# Frees all memory allocated by a GC heap.
function free!(heap::GCHeapDescription)
    for region in heap.regions
        CUDAdrv.Mem.free(region)
    end
end

# A sorted list of all allocation records for allocated blocks.
# This data structure is primarily useful for rapidly mapping
# pointers to the blocks allocated blocks that contain them.
struct SortedAllocationList
    # An array of pointers to allocation records. The pointers
    # are all sorted.
    records::Array{Ptr{FreeListRecord}, 1}
end

length(alloc_list::SortedAllocationList) = length(alloc_list.records)

# Gets a pointer to the allocation record that manages the memory
# pointed to by `pointer`. Returns a null pointer if there is no
# such record.
function get_record(
    alloc_list::SortedAllocationList,
    pointer::Ptr{T})::Ptr{FreeListRecord} where T

    # Deal with these cases quickly so we can assume that the
    # free list is nonempty.
    if length(alloc_list) == 0 ||
        pointer < data_pointer(alloc_list.records[1]) ||
        pointer >= data_end_pointer(alloc_list.records[end])

        return C_NULL
    end

    # To quickly narrow down the search space, we will do a binary search
    # for the biggest allocation record pointer that is smaller than `pointer`.
    range_start, range_end = 1, length(alloc_list)
    while range_end - range_start > 4
        range_mid = div(range_start + range_end, 2)
        mid_val = alloc_list.records[range_mid]
        if mid_val > pointer
            range_end = range_mid
        else
            range_start = range_mid
        end
    end

    # Make sure that the pointer actually points to a region of memory
    # that is managed by the candidate record we found.
    for record in alloc_list.records[range_start:range_end]
        if pointer >= data_pointer(record) && pointer < data_end_pointer(record)
            return record
        end
    end
    return C_NULL
end

# Iterates through a linked list of allocation records and apply a function
# to every node in the linked list.
function iterate_allocation_records(fun::Function, head::Ptr{FreeListRecord})
    while head != C_NULL
        fun(head)
        head = unsafe_load(head).next
    end
end

# Iterates through all active allocation records in a GC arena.
function iterate_allocated(fun::Function, arena::Ptr{FreeListArena})
    allocation_list_head = unsafe_load(arena).allocation_list_head
    iterate_allocation_records(fun, allocation_list_head)
end

# Iterates through all free allocation records in a GC arena.
function iterate_free(fun::Function, arena::Ptr{FreeListArena})
    free_list_head = unsafe_load(arena).free_list_head
    iterate_allocation_records(fun, free_list_head)
end

# Takes a GC master record and constructs a sorted allocation list
# based on it.
function sort_allocation_list(master_record::GCMasterRecord)::SortedAllocationList
    records = []
    iterate_arenas(master_record) do arena
        iterate_allocated(arena) do record
            push!(records, record)
        end
    end
    sort!(records)
    return SortedAllocationList(records)
end

# Frees all dead blocks in an arena.
function gc_free_garbage(arena::Ptr{FreeListArena}, live_blocks::Set{Ptr{FreeListRecord}})
    record_ptr = @get_field_pointer(arena, :allocation_list_head)
    while true
        record = unsafe_load(record_ptr)
        if record == C_NULL
            # We've reached the end of the list.
            break
        end

        if record in live_blocks
            # We found a live block. Proceed to the next block.
            record_ptr = @get_field_pointer(record, :next)
        else
            # We found a dead block. Release it. Don't proceed to the
            # next block because the current block will change in the
            # next iteration of this loop.
            gc_free_local(arena, record_ptr)
        end
    end
end

# Compact a GC arena's free list. This function will
#   1. merge adjancent free blocks, and
#   2. reorder free blocks to put small blocks at the front
#      of the free list,
#   3. tally the total number of free bytes and return that number.
function gc_compact(arena::Ptr{FreeListArena})::Csize_t
    # Let's start by creating a list of all free list records.
    records = Ptr{FreeListRecord}[]
    iterate_free(arena) do record
        push!(records, record)
    end

    # We now sort those records and loop through the sorted list,
    # merging free list entries as we go along.
    sort!(records)

    i = 1
    while i < length(records)
        first_record = records[i]
        second_record = records[i + 1]
        if data_end_pointer(first_record) == Base.unsafe_convert(Ptr{UInt8}, second_record)
            # We found two adjacent free list entries. Expand the first
            # record's size to encompass both entries, zero-fill the second
            # record's header and delete it from the list of records.
            new_size = Csize_t(data_end_pointer(second_record)) - Csize_t(data_pointer(first_record))
            zero_fill!(data_end_pointer(first_record), data_pointer(second_record))
            unsafe_store!(@get_field_pointer(first_record, :size), new_size)
            deleteat!(records, i + 1)
        else
            i += 1
        end
    end

    # Now sort the records based on size. Put the smallest records first to
    # discourage fragmentation.
    sort!(records; lt = (x, y) -> unsafe_load(x).size < unsafe_load(y).size)

    # Reconstruct the free list as a linked list.
    prev_record_ptr = @get_field_pointer(arena, :free_list_head)
    for record in records
        unsafe_store!(prev_record_ptr, record)
        prev_record_ptr = @get_field_pointer(record, :next)
    end
    unsafe_store!(prev_record_ptr, C_NULL)

    # Compute the total number of free bytes.
    return sum(map(record -> unsafe_load(record).size, records))
end

# Expands a GC arena by assigning it an additional heap region.
function gc_expand(arena::Ptr{FreeListArena}, region::GCHeapRegion)
    extra_record = make_gc_block!(pointer(region), Csize_t(sizeof(region)))
    last_free_list_ptr = @get_field_pointer(arena, :free_list_head)
    iterate_free(arena) do record
        last_free_list_ptr = @get_field_pointer(record, :next)
    end
    unsafe_store!(last_free_list_ptr, extra_record)
end

"""A report of the GC's actions."""
mutable struct GCReport
    """The total wall-clock time of a kernel execution."""
    elapsed_time::Float64

    """The number of collections that were performed."""
    collection_count::Int

    """The total wall-clock time of all collection polls."""
    collection_poll_time::Float64

    """The total wall-clock time of all collections."""
    collection_time::Float64

    """The total amount of additional memory allocated to local pools."""
    extra_local_memory::Csize_t

    """The total amount of additional memory allocated to the global pool."""
    extra_global_memory::Csize_t

    GCReport() = new(0.0, 0, 0.0, 0.0, Csize_t(0), Csize_t(0))
end

function show(io::IO, report::GCReport)
    print(io, "[wall-clock time: $(@sprintf("%.4f", report.elapsed_time)) s; ")
    print(io, "collections: $(report.collection_count); ")
    poll_percentage = 100 * report.collection_poll_time / report.elapsed_time
    print(io, "total poll time: $(@sprintf("%.4f", report.collection_poll_time)) s ($(@sprintf("%.2f", poll_percentage))%); ")
    collection_percentage = 100 * report.collection_time / report.elapsed_time
    print(io, "total collection time: $(@sprintf("%.4f", report.collection_time)) s ($(@sprintf("%.2f", collection_percentage))%); ")
    print(io, "extra local memory: $(div(report.extra_local_memory, MiB)) MiB; ")
    print(io, "extra global memory: $(div(report.extra_global_memory, MiB)) MiB]")
end

# Collects garbage. This function is designed to be called by the host,
# not by the device.
function gc_collect_impl(master_record::GCMasterRecord, heap::GCHeapDescription, config::GCConfiguration, report::GCReport)
    poll_time = Base.@elapsed begin
        # First off, we have to wait for all warps to reach a safepoint. Clear
        # safepoint flags and wait for warps to set them again.
        for i in 0:(master_record.warp_count - 1)
            atomic_compare_exchange!(
                master_record.safepoint_flags + i * sizeof(SafepointState),
                in_safepoint,
                not_in_safepoint)
        end
        safepoint_count = 0
        while safepoint_count != master_record.warp_count
            safepoint_count = 0
            for i in 0:(master_record.warp_count - 1)
                state = volatile_load(master_record.safepoint_flags + i * sizeof(SafepointState))
                if state != not_in_safepoint
                    safepoint_count += 1
                end
            end
        end
    end

    collection_time = Base.@elapsed begin

        # The Julia CPU GC is precise and the information it uses for precise
        # garbage collection is stored in memory that we should be able to access.
        # However, the way the CPU GC stores field information is incredibly
        # complicated and replicating that logic here would be a royal pain to
        # implement and maintain. Ideally, the CPU GC would expose an interface that
        # allows us to point to an object and ask the GC for all GC-tracked pointers
        # it contains. Alas, no such luck: the CPU GC doesn't even have an internal
        # function that does that. The CPU GC's logic for finding GC-tracked pointer
        # fields is instead fused tightly with its 'mark' loop.
        #
        # To cope with this, we will simply implement a semi-conservative GC: we precisely
        # scan the roots for pointers into the GC heap. We then recursively mark blocks
        # that are pointed to by such pointers as live and conservatively scan them for
        # more pointers.
        #
        # Our mark phase is fairly simple: we maintain a worklist of pointers that
        # are live and may need to be processed, as well as a set of blocks that are
        # live and have already been processed.
        live_blocks = Set{Ptr{FreeListRecord}}()
        live_worklist = Ptr{ObjectRef}[]

        # Get a sorted allocation list, which will allow us to classify live pointers quickly.
        alloc_list = sort_allocation_list(master_record)

        # Add all roots to the worklist.
        for i in 1:(master_record.root_buffer_capacity * master_record.thread_count)
            root = unsafe_load(master_record.root_buffers, i)
            if root != C_NULL
                push!(live_worklist, root)
            end
        end

        # Now process all live pointers until we reach a fixpoint.
        while !isempty(live_worklist)
            # Pop a pointer from the worklist.
            object_ref = pop!(live_worklist)
            # Get the block for that pointer.
            record = get_record(alloc_list, object_ref)
            # Make sure that we haven't visited the block yet.
            if record != C_NULL && !(record in live_blocks)
                # Mark the block as live.
                push!(live_blocks, record)
                # Add all pointer-sized, aligned values to the live pointer worklist.
                for ptr in data_pointer(record):sizeof(ObjectRef):data_end_pointer(record) - 1
                    value = unsafe_load(Base.unsafe_convert(Ptr{ObjectRef}, ptr))
                    push!(live_worklist, value)
                end
            end
        end

        # We're done with the mark phase! Time to proceed to the sweep phase.
        # The first thing we'll do is iterate through every arena's allocation list and
        # free dead blocks. Next, we will compact and reorder free lists to combat
        # fragmentation.
        iterate_arenas(master_record) do arena
            # Free garbage blocks.
            gc_free_garbage(arena, live_blocks)

            # Compact the arena.
            free_memory = gc_compact(arena)

            # If the amount of free memory in the arena is below the starvation
            # limit then we'll expand the GC heap and add the additional memory
            # to the arena's free list.
            threshold = if arena == master_record.global_arena
                config.global_arena_starvation_threshold
            else
                config.local_arena_starvation_threshold
            end

            if free_memory < threshold
                region = expand!(heap, threshold)
                gc_expand(arena, region)

                if arena == master_record.global_arena
                    report.extra_global_memory += Csize_t(threshold)
                else
                    report.extra_local_memory += Csize_t(threshold)
                end
            end
        end
    end
    report.collection_count += 1
    report.collection_time += collection_time
    report.collection_poll_time += poll_time
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
