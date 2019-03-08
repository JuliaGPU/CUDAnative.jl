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
# mark-and-sweep GC that runs on the host. The device may trigger
# the GC via an interrupt.
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
# MISCELLANEOUS
#
# Some miscellaneous GPU-related GC implementation details:
#
#   * GC memory is shared by the host and device.
#   * Every thread gets a fixed region of memory for storing GC roots in.
#   * When the device runs out of GC memory, it requests an interrupt
#     to mark and sweep.

export @cuda_gc, gc_malloc, gc_collect

import Base: length

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

# Gets a pointer to the first byte of data managed by an allocation record.
function data_pointer(record::Ptr{GCAllocationRecord})::Ptr{UInt8}
    Base.unsafe_convert(Ptr{UInt8}, record) + sizeof(GCAllocationRecord)
end

# Gets a pointer to the first byte of data no longer managed by an allocation record.
function data_end_pointer(record::Ptr{GCAllocationRecord})::Ptr{UInt8}
    data_pointer(record) + unsafe_load(@get_field_pointer(record, :size))
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

# A GC frame is just a pointer to an array of Julia objects.
const GCFrame = Ptr{ObjectRef}

# A data structure that contains global GC info. This data
# structure is designed to be immutable: it should not be changed
# once the host has set it up.
struct GCMasterRecord
    # A pointer to the global GC arena.
    global_arena::Ptr{GCArenaRecord}

    # The maximum size of a GC root buffer, i.e., the maximum number
    # of roots per thread.
    root_buffer_capacity::UInt32

    # The number of threads.
    thread_count::UInt32

    # A pointer to a list of root buffer pointers that point to the
    # end of the root buffer for every thread.
    root_buffer_fingers::Ptr{Ptr{ObjectRef}}

    # A pointer to a list of buffers that can be used to store GC roots in.
    # These root buffers are partitioned into GC frames later on.
    root_buffers::Ptr{ObjectRef}
end

# Iterates through all arena pointers stored in a GC master record.
@inline function iterate_arenas(fun::Function, master_record::GCMasterRecord)
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

# Gets the thread ID of the current thread.
@inline function get_thread_id()
    return (blockIdx().x - 1) * blockDim().x + threadIdx().x
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
    data_address = data_pointer(entry)

    # Compute the end of the free memory chunk.
    end_address = data_address + entry_data.size

    # Compute the start address of the new free list entry. The data
    # prefixed by the block needs to be aligned to a 16-byte boundary,
    # but the block itself doesn't.
    new_data_address = align_to_boundary(data_address + bytesize)
    new_entry_address = new_data_address - sizeof(GCAllocationRecord)
    if new_entry_address < data_address + bytesize
        new_entry_address += gc_align
        new_data_address += gc_align
    end

    # If we can place a new entry just past the allocation, then we should
    # by all means do so.
    if new_data_address < end_address
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
    # Acquire the arena's lock.
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
        gc_frame = new_gc_frame(UInt32(1))
        unsafe_store!(gc_frame, Base.unsafe_convert(ObjectRef, result_ptr))
    end
    return result_ptr
end

"""
    gc_malloc(bytesize::Csize_t)::Ptr{UInt8}

Allocates a blob of memory that is managed by the garbage collector.
This function is designed to be called by the device.
"""
function gc_malloc(bytesize::Csize_t)::Ptr{UInt8}
    master_record = get_gc_master_record()

    # Try to malloc the object without host intervention.
    ptr = @nocollect gc_malloc_local(master_record.global_arena, bytesize)
    if ptr != C_NULL
        return ptr
    end

    # We're out of memory, which means that we need the garbage collector
    # to step in. Acquire the interrupt lock.
    ptr = writer_locked(get_interrupt_lock()) do
        # Try to allocate memory again. This is bound to fail for the
        # first thread that acquires the interrupt lock, but it is quite
        # likely to succeed if we are *not* in the first thread that
        # acquired the garbage collector lock.
        ptr2 = gc_malloc_local(master_record.global_arena, bytesize)

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
function gc_free_local_impl(
    arena::Ptr{GCArenaRecord},
    record_ptr::Ptr{Ptr{GCAllocationRecord}})

    record = unsafe_load(record_ptr)
    next_record_ptr = @get_field_pointer(record, :next)
    free_list_head_ptr = @get_field_pointer(arena, :free_list_head)

    # Remove the record from the allocation list.
    next_record = unsafe_load(next_record_ptr)
    unsafe_store!(record_ptr, next_record)

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

# The initial size of the GC heap, currently 16 MiB.
const initial_gc_heap_size = 16 * (1 << 20)

# The default capacity of a root buffer, i.e., the max number of
# roots that can be stored per thread. Currently set to
# 256 roots. That's 2 KiB of roots per thread.
const default_root_buffer_capacity = 256

# The point at which an arena is deemed to be starving, i.e.,
# it no longer contains enough memory to perform basic allocations.
# If an arena's free byte count stays below the arena starvation
# threshold after a collection phase, the collector will allocate
# additional memory to the arena such that it is no longer starving.
# The arena starvation threshold is currently set to 4 MiB.
const arena_starvation_threshold = 4 * (1 << 20)

# A description of a region of memory that has been allocated to the GC heap.
struct GCHeapRegion
    # A buffer that contains the GC region's bytes.
    buffer::Array{UInt8, 1}
    # A pointer to the first element in the region.
    start::Ptr{UInt8}
    # The region's size in bytes.
    size::Csize_t
end

GCHeapRegion(buffer::Array{UInt8, 1}) = GCHeapRegion(buffer, pointer(buffer, 1), Csize_t(length(buffer)))

# A description of all memory that has been allocated to the GC heap.
struct GCHeapDescription
    # A list of the set of regions that comprise the GC heap.
    regions::Array{GCHeapRegion, 1}
end

GCHeapDescription() = GCHeapDescription([])

# Initializes a GC heap and produces a master record.
function gc_init!(
    heap::GCHeapDescription,
    thread_count::Integer;
    root_buffer_capacity::Integer = default_root_buffer_capacity)::GCMasterRecord

    master_region = heap.regions[1]

    gc_memory_start_ptr = master_region.start
    gc_memory_end_ptr = master_region.start + master_region.size

    # Set up root buffers.
    fingerbuf_bytesize = sizeof(Ptr{ObjectRef}) * thread_count
    fingerbuf_ptr = Base.unsafe_convert(Ptr{Ptr{ObjectRef}}, gc_memory_start_ptr)
    rootbuf_bytesize = sizeof(ObjectRef) * root_buffer_capacity * thread_count
    rootbuf_ptr = Base.unsafe_convert(Ptr{ObjectRef}, fingerbuf_ptr + fingerbuf_bytesize)

    # Populate the root buffer fingers.
    for i in 1:thread_count
        unsafe_store!(fingerbuf_ptr, rootbuf_ptr + (i - 1) * sizeof(ObjectRef) * root_buffer_capacity, i)
    end

    # Compute a pointer to the start of the heap.
    heap_start_ptr = rootbuf_ptr + rootbuf_bytesize

    # Create a single free list entry.
    first_entry_ptr = make_gc_block!(
        heap_start_ptr + sizeof(GCArenaRecord),
        Csize_t(gc_memory_end_ptr) - Csize_t(heap_start_ptr) - sizeof(GCArenaRecord))

    # Set up the main GC data structure.
    global_arena = Base.unsafe_convert(Ptr{GCArenaRecord}, heap_start_ptr)
    unsafe_store!(
        global_arena,
        GCArenaRecord(0, first_entry_ptr, C_NULL))

    return GCMasterRecord(global_arena, root_buffer_capacity, UInt32(thread_count), fingerbuf_ptr, rootbuf_ptr)
end

# Takes a zero-filled region of memory and turns it into a block
# managed by the GC, prefixed with an allocation record.
function make_gc_block!(start_ptr::Ptr{T}, size::Csize_t)::Ptr{GCAllocationRecord} where T
    entry = Base.unsafe_convert(Ptr{GCAllocationRecord}, start_ptr)
    unsafe_store!(
        entry,
        GCAllocationRecord(
            Csize_t(start_ptr + size) - Csize_t(data_pointer(entry)),
            C_NULL))
    return entry
end

# Tells if a GC heap contains a particular pointer.
function contains(heap::GCHeapDescription, pointer::Ptr{T}) where T
    for region in heap.regions
        if pointer >= region.start && pointer < region.start + region.size
            return true
        end
    end
    return false
end

# Expands the GC heap by allocating a region of memory and adding it to
# the list of allocated regions. `size` describes the amount of bytes to
# allocate. Returns the allocated region.
function expand!(heap::GCHeapDescription, size::Integer)::GCHeapRegion
    buffer = alloc_shared_array((size,), UInt8(0))
    region = GCHeapRegion(buffer)
    push!(heap.regions, region)
    return region
end

# Frees all memory allocated by a GC heap.
function free!(heap::GCHeapDescription)
    for region in heap.regions
        free_shared_array(region.buffer)
    end
end

# A sorted list of all allocation records for allocated blocks.
# This data structure is primarily useful for rapidly mapping
# pointers to the blocks allocated blocks that contain them.
struct SortedAllocationList
    # An array of pointers to allocation records. The pointers
    # are all sorted.
    records::Array{Ptr{GCAllocationRecord}, 1}
end

length(alloc_list::SortedAllocationList) = length(alloc_list.records)

# Gets a pointer to the allocation record that manages the memory
# pointed to by `pointer`. Returns a null pointer if there is no
# such record.
function get_record(
    alloc_list::SortedAllocationList,
    pointer::Ptr{T})::Ptr{GCAllocationRecord} where T

    cast_ptr = Base.unsafe_convert(Ptr{GCAllocationRecord}, pointer)

    # Deal with the most common cases quickly.
    if length(alloc_list) == 0 ||
        pointer < data_pointer(alloc_list.records[1]) ||
        pointer > data_pointer(alloc_list.records[end]) + Base.unsafe_load(alloc_list.records[end]).size

        return C_NULL
    end

    # To do this lookup quickly, we will do a binary search for the
    # biggest allocation record pointer that is smaller than `pointer`.
    range_start, range_end = 1, length(alloc_list)
    while range_end - range_start > 1 
        range_mid = div(range_start + range_end, 2)
        mid_val = alloc_list.records[range_mid]
        if mid_val > cast_ptr
            range_end = range_mid
        else
            range_start = range_mid
        end
    end

    record = alloc_list.records[range_end]
    if record >= cast_ptr
        record = alloc_list.records[range_start]
    end

    # Make sure that the pointer actually points to a region of memory
    # that is managed by the candidate record we found.
    record_data_pointer = data_pointer(record)
    if cast_ptr >= record_data_pointer && cast_ptr < record_data_pointer + unsafe_load(record).size
        return record
    else
        return C_NULL
    end
end

# Iterates through a linked list of allocation records and apply a function
# to every node in the linked list. The function is allowed to modify allocation
# records.
@inline function iterate_allocation_records(fun::Function, head::Ptr{GCAllocationRecord})
    while head != C_NULL
        fun(head)
        head = unsafe_load(head).next
    end
end

# Takes a GC master record and constructs a sorted allocation list
# based on it.
function sort_allocation_list(master_record::GCMasterRecord)::SortedAllocationList
    records = []
    iterate_arenas(master_record) do arena
        allocation_list_head = unsafe_load(arena).allocation_list_head
        iterate_allocation_records(allocation_list_head) do record
            push!(records, record)
        end
    end
    sort!(records)
    return SortedAllocationList(records)
end

# Compact a GC arena's free list. This function will
#   1. merge adjancent free blocks, and
#   2. reorder free blocks to put small blocks at the front
#      of the free list,
#   3. tally the total number of free bytes and return that number.
function gc_compact_free_list(arena::Ptr{GCArenaRecord})::Csize_t
    # Let's start by creating a list of all free list records.
    records = Ptr{GCAllocationRecord}[]
    free_list_head = unsafe_load(arena).free_list_head
    iterate_allocation_records(free_list_head) do record
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
    return sum(record -> unsafe_load(record).size, records)
end

# Collects garbage. This function is designed to be called by the host,
# not by the device.
function gc_collect_impl(master_record::GCMasterRecord, heap::GCHeapDescription)
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
    live_blocks = Set{Ptr{GCAllocationRecord}}()
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
            block_pointer = data_pointer(record)
            block_size = unsafe_load(record).size
            for i in 0:sizeof(ObjectRef):(block_size - 1)
                push!(live_worklist, Base.unsafe_convert(ObjectRef, block_pointer + i))
            end
        end
    end

    # We're done with the mark phase! Time to proceed to the sweep phase.
    # The first thing we'll do is iterate through every arena's allocation list and
    # free dead blocks. Next, we will compact and reorder free lists to combat
    # fragmentation.
    iterate_arenas(master_record) do arena
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
                gc_free_local_impl(arena, record_ptr)
            end
        end

        # Compact the free list.
        free_memory = gc_compact_free_list(arena)

        # If the amount of free memory in the arena is below the starvation
        # limit then we'll expand the GC heap and add the additional memory
        # to the arena's free list.
        if free_memory < arena_starvation_threshold
            region = expand!(heap, arena_starvation_threshold)
            extra_record = make_gc_block!(region.start, region.size)
            last_free_list_ptr = @get_field_pointer(arena, :free_list_head)
            iterate_allocation_records(unsafe_load(last_free_list_ptr)) do record
                last_free_list_ptr = @get_field_pointer(record, :next)
            end
            unsafe_store!(last_free_list_ptr, extra_record)
        end
    end
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
                local host_interrupt_array = alloc_shared_array((1,), ready)
                local device_interrupt_buffer = get_shared_device_buffer(host_interrupt_array)

                # Allocate a shared buffer for GC memory.
                local gc_memory_size = initial_gc_heap_size + sizeof(ObjectRef) * default_root_buffer_capacity * $(esc(thread_count))
                local gc_heap = GCHeapDescription()
                expand!(gc_heap, gc_memory_size)
                local master_record = gc_init!(gc_heap, $(esc(thread_count)))

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
                    gc_collect_impl(master_record, gc_heap)
                end

                try
                    # Standard kernel setup logic.
                    local kernel_args = CUDAnative.cudaconvert.(($(var_exprs...),))
                    local kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
                    local kernel = CUDAnative.cufunction($(esc(f)), kernel_tt; gc = true, $(map(esc, compiler_kwargs)...))
                    CUDAnative.prepare_kernel(kernel; init=kernel_init, $(map(esc, env_kwargs)...))
                    kernel(kernel_args...; $(map(esc, call_kwargs)...))

                    # Handle interrupts.
                    handle_interrupts(handle_interrupt, pointer(host_interrupt_array, 1), $(esc(stream)))
                finally
                    free_shared_array(host_interrupt_array)
                    free!(gc_heap)
                end
            end
         end)
    return code
end
