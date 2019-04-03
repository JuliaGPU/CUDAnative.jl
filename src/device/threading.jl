# This file implements threading primitives that work for CUDAnative kernels.

export ReaderWriterLock, reader_locked, writer_locked, Mutex, try_lock, unlock

# Gets a pointer to a global with a particular name. If the global
# does not exist yet, then it is declared in the global memory address
# space.
@generated function atomic_compare_exchange!(ptr::Ptr{T}, cmp::T, new::T)::T where T
    ptr_type = convert(LLVMType, Ptr{T})
    lt = string(convert(LLVMType, T))
    ir = """
        %ptr = inttoptr $ptr_type %0 to $lt*
        %result = cmpxchg volatile $lt* %ptr, $lt %1, $lt %2 seq_cst seq_cst
        %rv = extractvalue { $lt, i1 } %result, 0
        ret $lt %rv
        """
    :(Core.Intrinsics.llvmcall($ir, $T, Tuple{$(Ptr{T}), $T, $T}, ptr, cmp, new))
end

@generated function atomic_rmw!(::Val{op}, lhs::Ptr{T}, rhs::T)::T where {op, T}
    ptr_type = convert(LLVMType, Ptr{T})
    lt = string(convert(LLVMType, T))
    ir = """
        %ptr = inttoptr $ptr_type %0 to $lt*
        %rv = atomicrmw volatile $(String(op)) $lt* %ptr, $lt %1 seq_cst
        ret $lt %rv
        """
    :(Core.Intrinsics.llvmcall($ir, $T, Tuple{$(Ptr{T}), $T}, lhs, rhs))
end

# Atomically adds a value to a variable pointed to by a pointer.
# Returns the previous value stored in that variable.
function atomic_add!(lhs::Ptr{T}, rhs::T)::T where T
    atomic_rmw!(Val(:add), lhs, rhs)
end

# Atomically subtracts a value from a variable pointed to by a pointer.
# Returns the previous value stored in that variable.
function atomic_subtract!(lhs::Ptr{T}, rhs::T)::T where T
    atomic_rmw!(Val(:sub), lhs, rhs)
end

# Atomically computes the logical or of a value and a variable pointed
# to by a pointer. Returns the previous value stored in that variable.
function atomic_or!(lhs::Ptr{T}, rhs::T)::T where T
    atomic_rmw!(Val(:or), lhs, rhs)
end

# Atomically assigns a new value to a variable pointed to by a pointer.
# Returns the previous value stored in that variable.
function atomic_exchange!(lhs::Ptr{T}, rhs::T)::T where T
    atomic_rmw!(Val(:xchg), lhs, rhs)
end

# Loads a value from a pointer.
@generated function volatile_load(ptr::Ptr{T})::T where T
    ptr_type = string(convert(LLVMType, Ptr{T}))
    lt = string(convert(LLVMType, T))
    ir = """
        %ptr = inttoptr $ptr_type %0 to $lt*
        %rv = load volatile $lt, $lt* %ptr
        ret $lt %rv
        """
    :(Core.Intrinsics.llvmcall($ir, $T, Tuple{$(Ptr{T})}, ptr))
end

# Stores a value at a particular address.
@generated function volatile_store!(ptr::Ptr{T}, value::T) where T
    ptr_type = string(convert(LLVMType, Ptr{T}))
    lt = string(convert(LLVMType, T))
    ir = """
        %ptr = inttoptr $ptr_type %0 to $lt*
        store volatile $lt %1, $lt* %ptr
        ret void
        """
    :(Core.Intrinsics.llvmcall($ir, Cvoid, Tuple{$(Ptr{T}), $T}, ptr, value))
end

function unwrap_device_ptr(ptr::DevicePtr{T, A})::Ptr{T} where {T, A}
    convert(Ptr{T}, convert(Csize_t, ptr))
end

const ReaderWriterLockState = Int64

"""
A reader-writer lock: a lock that supports concurrent access for
read operations and exclusive access for write operations.
"""
struct ReaderWriterLock
    # A pointer to the reader-writer lock's state. The state
    # is a counter that can be in one of the following states:
    #
    #   * > 0: the lock is acquired by one or more readers.
    #          The state counter describes the number of readers
    #          that have acquired the lock.
    #
    #   * = 0: the lock is idle.
    #
    #   * < 0: the lock is acquired by a single writer.
    #
    state_ptr::Ptr{ReaderWriterLockState}
end

ReaderWriterLock(state_ptr::DevicePtr{ReaderWriterLockState}) =
    ReaderWriterLock(unwrap_device_ptr(state_ptr))

const max_rw_lock_readers = (1 << (sizeof(ReaderWriterLockState) * 8 - 1))

# Serializes execution of a function within a warp, to combat thread
# divergence-related deadlocks.
function warp_serialized(func::Function)
    # Get the current thread's ID.
    thread_id = threadIdx().x - 1

    # Get the size of a warp.
    size = warpsize()

    local result
    i = 0
    while i < size
        if thread_id % size == i
            result = func()
        end
        i += 1
    end
    return result
end

"""
    reader_locked(func::Function, lock::ReaderWriterLock; acquire_lock=true)

Acquires a reader-writer lock in reader mode, runs `func` while the lock is
acquired and releases the lock again.
"""
function reader_locked(func::Function, lock::ReaderWriterLock; acquire_lock=true)
    if !acquire_lock
        return func()
    end

    while true
        # Increment the reader count. If the lock is in write-acquired mode,
        # then the lock will stay in that mode (unless the reader count is
        # exceeded, but that is virtually impossible). Otherwise, the lock
        # will end up in read-acquired mode.
        previous_state = atomic_add!(lock.state_ptr, 1)

        # If the lock was in the idle or read-acquired state, then
        # it is now in read-acquired mode.
        if previous_state >= 0
            # Run the function.
            result = func()
            # Decrement the reader count to release the reader lock.
            atomic_add!(lock.state_ptr, -1)
            # We're done here.
            return result
        end

        # Decrement the reader count and try again.
        atomic_add!(lock.state_ptr, -1)
    end
end

"""
    writer_locked(func::Function, lock::ReaderWriterLock; acquire_lock=true)

Acquires a reader-writer lock in writer mode, runs `func` while the lock is
acquired and releases the lock again.
"""
function writer_locked(func::Function, lock::ReaderWriterLock; acquire_lock=true)
    if !acquire_lock
        return func()
    end

    warp_serialized() do
        # Try to move the lock from 'idle' to 'write-acquired'.
        while atomic_compare_exchange!(lock.state_ptr, 0, -max_rw_lock_readers) != 0
        end

        # We acquired the lock. Run the function.
        result = func()

        # Release the lock by atomically adding `max_rw_lock_readers` to the
        # lock's state. It's important that we use an atomic add instead of a
        # simple store because a store might cause a race condition with `read_locked`
        # that'll put us in a deadlock state.
        atomic_add!(lock.state_ptr, max_rw_lock_readers)

        # We're done here.
        return result
    end
end

# Gets the thread ID of the current thread.
@inline function get_thread_id()
    return (blockIdx().x - 1) * blockDim().x + threadIdx().x
end

# Gets the warp ID of the current thread.
@inline function get_warp_id()
    return div(get_thread_id() - 1, warpsize()) + 1
end

const MutexState = UInt32

"""
A mutex: a lock that guarantees mutual exclusion.
"""
struct Mutex
    # This GPU mutex implementation is based on
    # Lock-based Synchronization for GPU Architectures
    # by Yunlong Xu et al.
    state_ptr::Ptr{MutexState}
end

Mutex(state_ptr::DevicePtr{MutexState}) = 
    Mutex(unwrap_device_ptr(state_ptr))

"""
    unlock(mutex::Mutex)

Unlocks a mutex.
"""
function unlock(mutex::Mutex)
    threadfence()
    tid = get_thread_id()
    atomic_compare_exchange!(mutex.state_ptr, UInt32((tid << 1) + 1), UInt32(0))
    return
end

"""
    try_lock(mutex::Mutex)::Bool

Tries to acquire a lock on a mutex. Returns `true`
if a lock was acquired successfully; otherwise, `false`.
"""
function try_lock(mutex::Mutex)::Bool
    tid = UInt32(get_thread_id())
    wsize = warpsize()
    threadbit = UInt32(1) << (tid % wsize)

    mask = vote_ballot(true)

    bitset = @cuStaticSharedMem(UInt32, 128)
    bitset_ptr = unwrap_device_ptr(pointer(bitset)) + sizeof(UInt32) * div(threadIdx().x - 1, wsize)
    unsafe_store!(bitset_ptr, UInt32(0))

    lock = atomic_or!(mutex.state_ptr, UInt32(1))
    if lock & UInt32(1) == UInt32(0)
        # The lock is free.
        atomic_exchange!(mutex.state_ptr, UInt32((tid << 1) + 1))
    else
        pre_owner = lock >> 1
        if pre_owner != tid
            if div(lock, wsize << 1) == div(tid, wsize) && pre_owner > tid && (((mask >> (pre_owner % wsize)) & UInt32(1)) == UInt32(1))
                atomic_or!(bitset_ptr, UInt32(1 << (pre_owner % wsize)))
                atomic_exchange!(mutex.state_ptr, UInt32((tid << 1) + 1))
                if (atomic_or!(mutex.state_ptr, UInt32(0)) >> 1) != tid
                    # Stealing failed.
                    atomic_or!(bitset_ptr, threadbit)
                end
            else
                # Cannot steal.
                atomic_or!(bitset_ptr, threadbit)
            end
        end
    end

    if (unsafe_load(bitset_ptr) & threadbit) == UInt32(0)
        threadfence()
        return true
    else
        atomic_compare_exchange!(mutex.state_ptr, (tid << 1) + UInt32(1), UInt32(0))
        threadfence()
        return false
    end
end
