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

export @cuda_gc, gc_malloc, gc_malloc_object, gc_collect, gc_safepoint

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

# A data structure that describes a ScatterAlloc superblock. Every
# superblock is prefixed by one of these.
struct ScatterAllocSuperblock
    # The number of regions in the superblock.
    region_count::UInt32

    # The number of pages in a region managed by this superblock.
    pages_per_region::UInt32

    # The size of a page in the superblock, in bytes. This size
    # does not include the page's header.
    page_size::UInt32

    # A pointer to the next superblock.
    next::Ptr{ScatterAllocSuperblock}
end

# A region in a ScatterAlloc superblock.
struct ScatterAllocRegion
    # The number of pages in this region that are full.
    full_page_count::Int64
end

# A page in a ScatterAlloc region.
struct ScatterAllocPage
    # The size of a chunk in this page.
    chunk_size::Int64

    # The number of allocated blocks in this page.
    allocated_chunk_count::Int64

    # A bitmask that describes which chunks have been allocated
    # and which chunks are still free.
    occupancy::Int64
end

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

# Gets the page size in a superblock. This size does not include
# the page header.
function page_size(superblock::Ptr{ScatterAllocSuperblock})
    unsafe_load(@get_field_pointer(superblock, :page_size))
end

# Gets the number of pages per region in a superblock.
function pages_per_region(superblock::Ptr{ScatterAllocSuperblock})
    unsafe_load(@get_field_pointer(superblock, :pages_per_region))
end

# Gets the size of an aligned header, including padding to satisfy
# alignment requirements.
@generated function header_size(::Type{T}, ::Val{alignment} = Val(gc_align))::UInt32 where {T, alignment}
    result = align_upward(UInt32(sizeof(T)), alignment)
    :($result)
end

# Gets the total number of chunks in a particular page.
function chunk_count(page::Ptr{ScatterAllocPage}, superblock::Ptr{ScatterAllocSuperblock})
    chunk_size = unsafe_load(@get_field_pointer(page, :chunk_size))
    div(page_size(superblock), chunk_size)
end

# Gets the address of a particular chunk in a page. `index` is zero-based.
function chunk_address(page::Ptr{ScatterAllocPage}, index::Integer)::Ptr{UInt8}
    chunk_size = unsafe_load(@get_field_pointer(page, :chunk_size))
    Base.unsafe_convert(Ptr{UInt8}, page + header_size(ScatterAllocPage) + chunk_size * index)
end

# Gets the address of a particular page in a region. `index` is zero-based.
function page_address(region::Ptr{ScatterAllocRegion}, superblock::Ptr{ScatterAllocSuperblock}, index::Integer)::Ptr{ScatterAllocPage}
    Base.unsafe_convert(
        Ptr{ScatterAllocPage},
        region + header_size(ScatterAllocRegion) + index * (header_size(ScatterAllocPage) + page_size(superblock)))
end

# Gets the total size in bytes of a region, including overhead.
function region_bytesize(pages_per_region::Integer, page_size::Integer)
    region_data_size = pages_per_region * (header_size(ScatterAllocPage) + page_size)
    header_size(ScatterAllocRegion) + region_data_size
end

# Gets the address of a particular region in a superblock. `index` is zero-based.
function region_address(superblock::Ptr{ScatterAllocSuperblock}, index::Integer)::Ptr{ScatterAllocRegion}
    Base.unsafe_convert(
        Ptr{ScatterAllocPage},
        superblock + header_size(ScatterAllocSuperblock) + index * region_bytesize(pages_per_region(superblock), page_size(superblock)))
end

# A GC arena that uses the ScatterAlloc algorithm for allocations.
struct ScatterAllocArena
    # A pointer to the first superblock managed by this arena.
    first_superblock::Ptr{ScatterAllocSuperblock}
end

# A "shelf" in a bodega arena. See `BodegaArena` for more info on
# how shelves work.
struct BodegaShelf
    # The size of the chunks on this shelf.
    chunk_size::Csize_t

    # The maximal number of chunks on this shelf.
    capacity::Int64

    # An index into the shelf that points to the first free
    # chunk. This is a zero-based index.
    chunk_finger::Int64

    # A pointer to an array of pointers to chunks of memory.
    # Every chunk in this array has a chunk size that is
    # at least as large as `chunk_size`.
    chunks::Ptr{Ptr{UInt8}}
end

# A GC arena that uses a custom ("bodega") allocation algorithm for allocations.
# Essentially, this type of arena has a list of "shelves" that contain small,
# preallocated chunks of memory that threads can claim in a fast and lock-free
# manner. When the shelves run out of memory, threads may re-stock them from free
# list, amortizing the cost of lock acquisition across many different allocations.
struct BodegaArena
    # The number of shelves in the arena.
    shelf_count::Int

    # A pointer to an array of shelves.
    shelves::Ptr{BodegaShelf}

    # A Boolean that tells if it is sensible to try and restock shelves in this
    # arena. Restocking shelves becomes futile once the free list's capacity is
    # exhausted.
    can_restock::Bool

    # The free list this bodega uses for large allocations and for re-stocking
    # the shelves.
    free_list::FreeListArena
end

# Gets a pointer to a bodega arena's free list.
function get_free_list(arena::Ptr{BodegaArena})::Ptr{FreeListArena}
    @get_field_pointer(arena, :free_list)
end

# Gets the first shelf containing chunks that are at least `bytesize` bytes
# in size. Returns null if there is no such shelf.
function get_shelf(arena::Ptr{BodegaArena}, bytesize::Csize_t)::Ptr{BodegaShelf}
    bodega = unsafe_load(arena)
    for i in 1:bodega.shelf_count
        shelf = bodega.shelves + (i - 1) * sizeof(BodegaShelf)
        chunk_size = unsafe_load(@get_field_pointer(shelf, :chunk_size))
        if chunk_size >= bytesize
            return shelf
        end
    end
    return C_NULL
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

    # A pointer to the tiny arena, which uses the ScatterAlloc
    # algorithm to provision space for small objects.
    tiny_arena::Ptr{ScatterAllocArena}

    # A pointer to a list of local GC arena pointers.
    local_arenas::Ptr{Ptr{BodegaArena}}

    # A pointer to the global GC arena.
    global_arena::Ptr{BodegaArena}

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
@inline function get_local_arena()::Ptr{BodegaArena}
    master_record = get_gc_master_record()
    if master_record.local_arena_count == UInt32(0)
        return C_NULL
    else
        return unsafe_load(
            master_record.local_arenas,
            get_warp_id() % master_record.local_arena_count)
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
        gc_set_safepoint_flag(in_safepoint)
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
function gc_set_safepoint_flag(value::SafepointState)
    master_record = get_gc_master_record()
    warp_id = get_warp_id()
    safepoint_flag_ptr = master_record.safepoint_flags + sizeof(SafepointState) * (warp_id - 1)
    volatile_store!(safepoint_flag_ptr, value)
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

# Tries to use a free-list entry to allocate a chunk of data of size `bytesize`.
# Updates the free list if the allocation succeeds. Returns a null pointer otherwise.
function gc_use_free_list_entry(
    entry_ptr::Ptr{Ptr{FreeListRecord}},
    allocation_list_ptr::Ptr{Ptr{FreeListRecord}},
    entry::Ptr{FreeListRecord},
    bytesize::Csize_t)::Ptr{UInt8}

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

    # At this point, all we need to do is update the allocation record to
    # reflect the fact that it now represents an allocated block instead of
    # a free block.

    # Set the `next` pointer to the value stored at the allocation list pointer.
    unsafe_store!(
        @get_field_pointer(entry, :next)::Ptr{Ptr{FreeListRecord}},
        unsafe_load(allocation_list_ptr))

    # Update the allocation list pointer to point to the entry.
    unsafe_store!(allocation_list_ptr, entry)

    return data_address
end

# Tries to allocate a chunk of memory from a ScatterAlloc page.
# Returns a null pointer if no chunk of memory can be found.
function gc_scatter_alloc_use_page(
    page::Ptr{ScatterAllocPage},
    region::Ptr{ScatterAllocRegion},
    superblock::Ptr{ScatterAllocSuperblock})::Ptr{UInt8}

    alloc_chunk_ptr = @get_field_pointer(page, :allocated_chunk_count)
    fill_level = atomic_add!(alloc_chunk_ptr, 1)
    spots = chunk_count(page, superblock)
    if fill_level < spots
        if fill_level + 1 == spots
            # The page is full now. Increment the region's counter.
            full_page_ptr = @get_field_pointer(region, :full_page_count)
            atomic_add!(full_page_ptr, 1)
        end

        lane_id = (get_thread_id() - 1) % warpsize()
        spot = lane_id % spots
        occupancy_ptr = @get_field_pointer(page, :occupancy)
        while true
            # Check if our preferred spot is available.
            mask = 1 << spot
            old = atomic_or!(occupancy_ptr, mask)

            actual_fill = 0
            for i in 1:64
                if old & (1 << (i - 1)) != 0
                    actual_fill += 1
                end
            end

            # If the spot is available, then use it.
            if old & mask == 0
                break
            end

            # Otherwise, find a new spot.
            spot = (spot + 1) % spots
        end
        return chunk_address(page, spot)
    end

    # The page is full.
    atomic_subtract!(alloc_chunk_ptr, 1)
    return C_NULL
end

function scatter_alloc_hash(
    superblock::Ptr{ScatterAllocSuperblock},
    bytesize::Int64)::Int64

    sb = unsafe_load(superblock)
    page_count = sb.region_count * sb.pages_per_region
    warp_id = get_warp_id() - 1

    k_S = 38183
    k_mp = 17497

    (bytesize * k_S + warp_id * k_mp) % page_count
end

# Tries to allocate a chunk of memory from a ScatterAlloc superblock.
# Returns a null pointer if no sufficiently large chunk of
# memory can be found.
function gc_scatter_alloc_use_superblock(
    superblock::Ptr{ScatterAllocSuperblock},
    bytesize::Csize_t)::Ptr{UInt8}

    if bytesize > page_size(superblock)
        # This isn't going to work. The superblock's page size is just too small.
        return C_NULL
    end

    # Choose the allocation size in such a way that we never end up with more than
    # 64 chunks. This is necessary because the chunk occupancy bitfield is only
    # 64 bits wide.
    alloc_size = Int64(div(page_size(superblock), 64))
    if alloc_size < Int64(bytesize)
        alloc_size = Int64(bytesize)
    end

    # Align the allocation size.
    alloc_size = align_upward(alloc_size)

    # We are looking for a chunk that is `bytesize` bytes in size,
    # but we're willing to accept a chunk that is twice as large.
    waste_factor = 2
    max_size = alloc_size * waste_factor

    pages_per_region = unsafe_load(@get_field_pointer(superblock, :pages_per_region))
    region_count = unsafe_load(@get_field_pointer(superblock, :region_count))

    # Guess a global page index.
    global_page_id = scatter_alloc_hash(superblock, alloc_size)

    # Decompose that global page index into a region index and a
    # local page index.
    region_id = global_page_id % pages_per_region
    page_id = div(global_page_id, pages_per_region)

    # Remember the initial values of the region and page ids.
    init_region_id = region_id
    init_page_id = page_id

    # Find the region and page corresponding to the current page ID.
    region = region_address(superblock, region_id)
    while true
        page = page_address(region, superblock, page_id)

        # Skip regions until we find a region that is sufficiently empty.
        while true
            region_fill_level = unsafe_load(region).full_page_count / pages_per_region
            if region_fill_level > 0.9
                region_id += 1
                if region_id >= region_count
                    region_id = 0
                end
                region = region_address(superblock, region_id)
                page_id = 0
            else
                break
            end
        end

        # Try to set the chunk size to our preferred chunk size.
        chunk_size_ptr = @get_field_pointer(page, :chunk_size)
        chunk_size = atomic_compare_exchange!(chunk_size_ptr, 0, alloc_size)
        if chunk_size == 0 || (chunk_size >= alloc_size && chunk_size <= max_size)
            # If we managed to set the page's chunk size, then the page is definitely
            # suitable for our purposes. Otherwise, the page might still be suitable
            # if its chunk size is sufficiently large to accommodate the requested
            # size yet small enough to not waste too much space.
            result = gc_scatter_alloc_use_page(page, region, superblock)
            if result != C_NULL
                return result
            end
        end

        # Try the next page.
        page_id += 1

        if page_id >= pages_per_region
            region_id += 1
            if region_id >= region_count
                region_id = 0
            end
            region = region_address(superblock, region_id)
            page_id = 0
        end

        # We tried every page in the entire superblock and found nothing.
        if region_id == init_region_id && page_id == init_page_id
            return C_NULL
        end
    end
end

# Tries to allocate a chunk of memory in a particular GC arena.
# Returns a null pointer if no sufficiently large chunk of
# memory can be found.
function gc_malloc_local(arena::Ptr{ScatterAllocArena}, bytesize::Csize_t)::Ptr{UInt8}
    # Walk the list of superblocks until we find a valid candidate.
    superblock = unsafe_load(arena).first_superblock
    while superblock != C_NULL
        result = gc_scatter_alloc_use_superblock(superblock, bytesize)
        if result != C_NULL
            return result
        end
        superblock = unsafe_load(@get_field_pointer(superblock, :next))
    end

    return C_NULL
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
    free_list_ptr::Ptr{Ptr{FreeListRecord}},
    allocation_list_ptr::Ptr{Ptr{FreeListRecord}},
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

        free_list_ptr = @get_field_pointer(free_list_item, :next)::Ptr{Ptr{FreeListRecord}}
    end
    return C_NULL
end

# Tries to allocate a chunk of memory from a free list.
# Returns a null pointer if no sufficiently large chunk of
# memory can be found.
#
# This function is not thread-safe.
function gc_malloc_from_free_list(arena::Ptr{FreeListArena}, bytesize::Csize_t)::Ptr{UInt8}
    free_list_ptr = @get_field_pointer(arena, :free_list_head)::Ptr{Ptr{FreeListRecord}}
    allocation_list_ptr = @get_field_pointer(arena, :allocation_list_head)::Ptr{Ptr{FreeListRecord}}
    gc_malloc_from_free_list(free_list_ptr, allocation_list_ptr, bytesize)
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
function gc_malloc_local(arena::Ptr{FreeListArena}, bytesize::Csize_t)::Ptr{UInt8}
    # Acquire the arena's lock.
    result_ptr = writer_locked(get_lock(arena)) do
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

# Atomically takes a chunk from a shelf. Returns null if the shelf
# is empty.
function gc_malloc_from_shelf(shelf::Ptr{BodegaShelf})::Ptr{UInt8}
    capacity = unsafe_load(@get_field_pointer(shelf, :capacity))

    # Atomically increment the chunk finger.
    finger_ptr = @get_field_pointer(shelf, :chunk_finger)
    finger = atomic_add!(finger_ptr, 1)

    if finger < capacity
        # If the chunk finger was less than the capacity, then we actually
        # managed to take a chunk from the shelf. We only need to retrieve
        # its address.
        chunk_array = unsafe_load(@get_field_pointer(shelf, :chunks))
        return unsafe_load(chunk_array, finger + 1)
    else
        # Otherwise, we've got nothing. Return null.
        return C_NULL
    end
end

# Re-stocks a shelf.
function restock_shelf(arena::Ptr{BodegaArena}, shelf::Ptr{BodegaShelf})
    shelf_size = unsafe_load(@get_field_pointer(shelf, :chunk_size))
    capacity = unsafe_load(@get_field_pointer(shelf, :capacity))
    finger_ptr = @get_field_pointer(shelf, :chunk_finger)
    finger = unsafe_load(finger_ptr)

    # The finger may exceed the capacity. This is harmless. Just
    # reset the finger to the capacity.
    if finger > capacity
        finger = capacity
    end

    # Actually re-stock the shelf.
    free_list = get_free_list(arena)
    chunk_array = unsafe_load(@get_field_pointer(shelf, :chunks))
    while finger > 0
        chunk = gc_malloc_from_free_list(free_list, shelf_size)
        if chunk == C_NULL
            # We exhausted the free list. Better break now. Also set
            # the arena's `can_restock` flag to false so there will be
            # no future attempts to re-stock shelves.
            unsafe_store!(@get_field_pointer(arena, :can_restock), false)
            break
        end

        # Update the chunk array.
        unsafe_store!(chunk_array, chunk, finger)
        finger -= 1
    end

    # Update the finger.
    unsafe_store!(finger_ptr, finger)
end

# Tries to allocate a chunk of memory in a particular GC arena.
# Returns a null pointer if no sufficiently large chunk of
# memory can be found.
function gc_malloc_local(arena::Ptr{BodegaArena}, bytesize::Csize_t)::Ptr{UInt8}
    # The bodega arena might be empty (or approximately empty). If so, then we'll
    # just return null early. There's no need to scrape the bottom of the barrel.
    if !unsafe_load(@get_field_pointer(arena, :can_restock))
        return C_NULL
    end

    # Find the right shelf for this allocation.
    shelf = get_shelf(arena, bytesize)
    free_list = get_free_list(arena)
    if shelf == C_NULL
        # The shelves' chunk sizes are all too small to accommodate this
        # allocation. Use the free list directly.
        return gc_malloc_local(free_list, bytesize)
    end

    # Acquire a reader lock on the arena and try to take a chunk
    # from the shelf.
    lock = get_lock(free_list)
    result_ptr = reader_locked(lock) do
        gc_malloc_from_shelf(shelf)
    end

    if result_ptr == C_NULL
        # Looks like we need to re-stock the shelf. While we're at it,
        # we might as well grab a chunk of memory for ourselves.
        result_ptr = writer_locked(lock) do
            restock_shelf(arena, shelf)
            gc_malloc_from_free_list(free_list, bytesize)
        end
    end

    gc_protect(result_ptr)
    return result_ptr
end

"""
    gc_malloc(bytesize::Csize_t)::Ptr{UInt8}

Allocates a blob of memory that is managed by the garbage collector.
This function is designed to be called by the device.
"""
function gc_malloc(bytesize::Csize_t)::Ptr{UInt8}
    master_record = get_gc_master_record()

    function allocate()
        # Try to allocate in the tiny arena first. The ScatterAlloc
        # algorithm used by that arena is lock-free and works well
        # for small objects.
        if master_record.tiny_arena != C_NULL
            local_ptr = gc_malloc_local(master_record.tiny_arena, bytesize)
            if local_ptr != C_NULL
                return local_ptr
            end
        end

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
        end

        # Try to use the global arena if all else fails.
        gc_malloc_local(master_record.global_arena, bytesize)
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
        #
        # Note: don't try to allocate in the local arena first because
        # we have already acquired a device-wide lock. Allocating in
        # the local arena first might waste precious time.
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

# One megabyte.
const MiB = 1 << 20

# The initial size of the GC heap, currently 20 MiB.
const initial_gc_heap_size = 16 * MiB

# The default capacity of a root buffer, i.e., the max number of
# roots that can be stored per thread. Currently set to
# 256 roots. That's 2 KiB of roots per thread.
const default_root_buffer_capacity = 256

# The point at which the global arena is deemed to be starving, i.e.,
# it no longer contains enough memory to perform basic allocations.
# If the global arena's free byte count stays below the arena starvation
# threshold after a collection phase, the collector will allocate
# additional memory to the arena such that it is no longer starving.
# The arena starvation threshold is currently set to 4 MiB.
const global_arena_starvation_threshold = 4 * MiB

# The point at which a local arena is deemed to be starving, i.e.,
# it no longer contains enough memory to perform basic allocations.
# If a local arena's free byte count stays below the arena starvation
# threshold after a collection phase, the collector will allocate
# additional memory to the arena such that it is no longer starving.
# The arena starvation threshold is currently set to 1 MiB.
const local_arena_starvation_threshold = 1 * MiB

# The point at which a tiny arena is deemed to be starving, i.e.,
# it no longer contains enough memory to perform basic allocations.
# If a tiny arena's free byte count stays below the arena starvation
# threshold after a collection phase, the collector will allocate
# additional memory to the arena such that it is no longer starving.
# This arena starvation threshold is currently set to 2 MiB.
const tiny_arena_starvation_threshold = 0 # 2 * MiB

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
    warp_count::Union{Integer, Nothing} = nothing,
    root_buffer_capacity::Integer = default_root_buffer_capacity,
    local_arena_count::Integer = 8)::GCMasterRecord

    if warp_count == nothing
        warp_count = Base.ceil(UInt32, thread_count / CUDAdrv.warpsize(device()))
    end

    master_region = heap.regions[1]

    gc_memory_start_ptr = master_region.start
    gc_memory_end_ptr = master_region.start + master_region.size

    # Allocate a local arena pointer buffer.
    local_arenas_bytesize = sizeof(Ptr{BodegaArena}) * local_arena_count
    local_arenas_ptr = Base.unsafe_convert(Ptr{Ptr{BodegaArena}}, gc_memory_start_ptr)

    # Allocate the safepoint flag buffer.
    safepoint_bytesize = sizeof(SafepointState) * warp_count
    safepoint_ptr = Base.unsafe_convert(Ptr{SafepointState}, local_arenas_ptr + local_arenas_bytesize)

    # Allocate root buffers.
    fingerbuf_bytesize = sizeof(Ptr{ObjectRef}) * thread_count
    fingerbuf_ptr = Base.unsafe_convert(Ptr{Ptr{ObjectRef}}, safepoint_ptr + fingerbuf_bytesize)
    rootbuf_bytesize = sizeof(ObjectRef) * root_buffer_capacity * thread_count
    rootbuf_ptr = Base.unsafe_convert(Ptr{ObjectRef}, fingerbuf_ptr + fingerbuf_bytesize)

    # Populate the root buffer fingers.
    for i in 1:thread_count
        unsafe_store!(fingerbuf_ptr, rootbuf_ptr + (i - 1) * sizeof(ObjectRef) * root_buffer_capacity, i)
    end

    # Compute a pointer to the start of the tiny arena.
    arena_start_ptr = rootbuf_ptr + rootbuf_bytesize

    # Set up the tiny object arena.
    if tiny_arena_starvation_threshold > 0
        arena_for_ants = make_gc_arena!(ScatterAllocArena, arena_start_ptr, Csize_t(tiny_arena_starvation_threshold))
        arena_start_ptr += tiny_arena_starvation_threshold
    else
        arena_for_ants = Base.unsafe_convert(Ptr{ScatterAllocArena}, C_NULL)
    end

    # Set up local arenas.
    for i in 1:local_arena_count
        local_arena = make_gc_arena!(BodegaArena, arena_start_ptr, Csize_t(local_arena_starvation_threshold))
        unsafe_store!(local_arenas_ptr, local_arena, i)
        arena_start_ptr += local_arena_starvation_threshold
    end

    # Set up the global arena.
    global_arena = make_gc_arena!(BodegaArena, arena_start_ptr, Csize_t(gc_memory_end_ptr) - Csize_t(arena_start_ptr))

    return GCMasterRecord(
        warp_count,
        UInt32(thread_count),
        root_buffer_capacity,
        UInt32(local_arena_count),
        arena_for_ants,
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

# Takes a zero-filled region of memory and turns it into an arena
# managed by the GC, prefixed with an arena record.
function make_gc_arena!(::Type{BodegaArena}, start_ptr::Ptr{T}, size::Csize_t)::Ptr{BodegaArena} where T
    current_ptr = start_ptr + sizeof(BodegaArena)

    # Set up some shelf chunk arrays
    shelf_records = []
    for chunk_size in [32, 64]
        capacity = 2048
        shelf_chunk_array = Base.unsafe_convert(Ptr{Ptr{UInt8}}, current_ptr)
        current_ptr += capacity * sizeof(Ptr{UInt8})
        push!(shelf_records, BodegaShelf(Csize_t(chunk_size), capacity, capacity, shelf_chunk_array))
    end

    # Set up the shelves.
    shelf_array = Base.unsafe_convert(Ptr{BodegaShelf}, current_ptr)
    for record in shelf_records
        shelf = Base.unsafe_convert(Ptr{BodegaShelf}, current_ptr)
        current_ptr += sizeof(BodegaShelf)
        unsafe_store!(shelf, record)
    end

    # Set up a free list entry.
    first_entry_ptr = make_gc_block!(current_ptr, Csize_t(start_ptr + size) - Csize_t(current_ptr))

    # Set up the arena record.
    arena = Base.unsafe_convert(Ptr{BodegaArena}, start_ptr)
    unsafe_store!(
        arena,
        BodegaArena(
            length(shelf_records),
            shelf_array,
            true,
            FreeListArena(0, first_entry_ptr, C_NULL)))

    # Stock the shelves.
    for record in shelf_records
        restock_shelf(arena, get_shelf(arena, record.chunk_size))
    end

    arena
end

# Takes a zero-filled region of memory and turns it into a ScatterAlloc
# superblock.
function make_gc_superblock!(
    start_ptr::Ptr{T},
    size::Csize_t;
    page_size::UInt32 = UInt32(2048),
    pages_per_region::UInt32 = UInt32(16))::Ptr{ScatterAllocSuperblock} where T

    region_size = region_bytesize(pages_per_region, page_size)

    # Figure out how many regions we can allocate.
    region_count = div(size - header_size(ScatterAllocSuperblock), region_size)

    # At this point, we'd normally allocate regions and pages.
    # However, region and page headers are zero-initialized by default.
    # So we don't actually need to do anything to set up the regions
    # and pages.

    # Allocate the superblock header.
    superblock = Base.unsafe_convert(Ptr{ScatterAllocSuperblock}, align_upward(start_ptr))
    unsafe_store!(
        superblock,
        ScatterAllocSuperblock(region_count, pages_per_region, page_size, C_NULL))

    superblock
end

# Takes a zero-filled region of memory and turns it into an arena
# managed by the GC, prefixed with an arena record.
function make_gc_arena!(::Type{ScatterAllocArena}, start_ptr::Ptr{T}, size::Csize_t)::Ptr{ScatterAllocArena} where T
    superblock_ptr = align_upward(start_ptr + sizeof(ScatterAllocArena))
    superblock = make_gc_superblock!(superblock_ptr, Csize_t(start_ptr) + size - Csize_t(superblock_ptr))
    arena = Base.unsafe_convert(Ptr{ScatterAllocArena}, start_ptr)
    unsafe_store!(
        arena,
        ScatterAllocArena(superblock))

    arena
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
    records::Array{Ptr{FreeListRecord}, 1}
end

length(alloc_list::SortedAllocationList) = length(alloc_list.records)

# Gets a pointer to the allocation record that manages the memory
# pointed to by `pointer`. Returns a null pointer if there is no
# such record.
function get_record(
    alloc_list::SortedAllocationList,
    pointer::Ptr{T})::Ptr{FreeListRecord} where T

    cast_ptr = Base.unsafe_convert(Ptr{FreeListRecord}, pointer)

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

# Iterates through all active allocation records in a GC arena.
function iterate_allocated(fun::Function, arena::Ptr{BodegaArena})
    # Compose a set that contains all data addresses of chunks that
    # are on the shelves.
    arena_data = unsafe_load(arena)
    chunks_on_shelves = Set{Ptr{UInt8}}()
    for i in 1:arena_data.shelf_count
        shelf = unsafe_load(arena_data.shelves, i)
        for j in shelf.chunk_finger:(shelf.capacity - 1)
            push!(chunks_on_shelves, unsafe_load(shelf.chunks, j))
        end
    end

    # Now iterate through the allocation list, ignoring records that have
    # been placed on the shelves.
    iterate_allocated(get_free_list(arena)) do record
        if !(data_pointer(record) in chunks_on_shelves)
            fun(record)
        end
    end
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

# Frees all dead blocks in an arena.
function gc_free_garbage(arena::Ptr{BodegaArena}, live_blocks::Set{Ptr{FreeListRecord}})
    # Free garbage in the free list sub-arena.
    gc_free_garbage(get_free_list(arena), live_blocks)

    # Mark the arena as ready for restocking.
    unsafe_store!(@get_field_pointer(arena, :can_restock), true)
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
    return sum(record -> unsafe_load(record).size, records)
end

# Compact a GC arena's free list. This function will
#   1. merge adjancent free blocks, and
#   2. reorder free blocks to put small blocks at the front
#      of the free list,
#   3. tally the total number of free bytes and return that number.
function gc_compact(arena::Ptr{BodegaArena})::Csize_t
    # Compact the free list.
    tally = gc_compact(get_free_list(arena))

    # Add the size of the chunks on shelves to the tally.
    shelf_count = unsafe_load(@get_field_pointer(arena, :shelf_count))
    for i in 1:shelf_count
        shelf_array = unsafe_load(@get_field_pointer(arena, :shelves))
        shelf_data = unsafe_load(shelf_array, i)

        finger = shelf_data.chunk_finger
        if finger > shelf_data.capacity
            finger = shelf_data.capacity
        end
        tally += shelf_data.chunk_size * (shelf_data.capacity - finger)
    end

    tally
end

# Expands a GC arena by assigning it an additional heap region.
function gc_expand(arena::Ptr{FreeListArena}, region::GCHeapRegion)
    extra_record = make_gc_block!(region.start, region.size)
    last_free_list_ptr = @get_field_pointer(arena, :free_list_head)
    iterate_free(arena) do record
        last_free_list_ptr = @get_field_pointer(record, :next)
    end
    unsafe_store!(last_free_list_ptr, extra_record)
end

# Expands a GC arena by assigning it an additional heap region.
function gc_expand(arena::Ptr{BodegaArena}, region::GCHeapRegion)
    gc_expand(get_free_list(arena), region)
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
function gc_collect_impl(master_record::GCMasterRecord, heap::GCHeapDescription, report::GCReport)
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
            # Free garbage blocks.
            gc_free_garbage(arena, live_blocks)

            # Compact the arena.
            free_memory = gc_compact(arena)

            # If the amount of free memory in the arena is below the starvation
            # limit then we'll expand the GC heap and add the additional memory
            # to the arena's free list.
            threshold = if arena == master_record.global_arena
                global_arena_starvation_threshold
            else
                local_arena_starvation_threshold
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

"""
    @cuda_gc [kwargs...] func(args...)

High-level interface for executing code on a GPU with GC support.
The `@cuda_gc` macro should prefix a call, with `func` a callable function
or object that should return nothing. It will be compiled to a CUDA function upon first
use, and to a certain extent arguments will be converted and managed automatically using
`cudaconvert`. Next, a call to `CUDAdrv.cudacall` is performed, scheduling a kernel
launch on the current CUDA context. Finally, `@cuda_gc` waits for the kernel to finish,
performing garbage collection in the meantime if necessary.

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

                local gc_report = GCReport()
                local function handle_interrupt()
                    gc_collect_impl(master_record, gc_heap, gc_report)
                end

                try
                    # Standard kernel setup logic.
                    local kernel_args = CUDAnative.cudaconvert.(($(var_exprs...),))
                    local kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
                    local kernel = CUDAnative.cufunction($(esc(f)), kernel_tt; gc = true, $(map(esc, compiler_kwargs)...))
                    CUDAnative.prepare_kernel(kernel; init=kernel_init, $(map(esc, env_kwargs)...))
                    gc_report.elapsed_time = Base.@elapsed begin
                        kernel(kernel_args...; $(map(esc, call_kwargs)...))

                        # Handle interrupts.
                        handle_interrupts(handle_interrupt, pointer(host_interrupt_array, 1), $(esc(stream)))
                    end
                finally
                    free_shared_array(host_interrupt_array)
                    free!(gc_heap)
                end
                gc_report
            end
         end)
    return code
end
