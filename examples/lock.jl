using CUDAdrv, CUDAnative
using Test

thread_count = 128

# Define a kernel that atomically increments a counter using a lock.
function increment_counter(counter::CUDAnative.DevicePtr{Int32}, lock_state::CUDAnative.DevicePtr{CUDAnative.ReaderWriterLockState})
    lock = ReaderWriterLock(lock_state)
    writer_locked(lock) do
        unsafe_store!(counter, unsafe_load(counter) + 1)
    end
    return
end

# Allocate memory for the counter and the lock.
counter_buf = Mem.alloc(sizeof(Int32))
Mem.upload!(counter_buf, [Int32(0)])
counter_pointer = Base.unsafe_convert(CuPtr{Int32}, counter_buf)

lock_buf = Mem.alloc(sizeof(CUDAnative.ReaderWriterLockState))
Mem.upload!(lock_buf, [CUDAnative.ReaderWriterLockState(0)])
lock_pointer = Base.unsafe_convert(CuPtr{CUDAnative.ReaderWriterLockState}, lock_buf)

# @device_code_warntype increment_counter(counter_pointer, lock_pointer)

# Run the kernel.
@cuda threads=thread_count increment_counter(counter_pointer, lock_pointer)

# Check that the counter's final value equals the number
# of threads.
@test Mem.download(Int32, counter_buf) == [Int32(thread_count)]
