using CUDAdrv, CUDAnative
using Test

const thread_count = Int32(128)
const total_count = Int32(1024)

# Define a kernel that atomically increments a counter using a lock.
function increment_counter(counter::CUDAnative.DevicePtr{Int32}, lock_state::CUDAnative.DevicePtr{CUDAnative.MutexState})
    lock = Mutex(lock_state)
    done = false
    while !done && try_lock(lock)
        new_count = unsafe_load(counter) + 1
        unsafe_store!(counter, new_count)
        if new_count == total_count
            done = true
        end
        CUDAnative.unlock(lock)
    end
    return
end

# Allocate memory for the counter and the lock.
counter_buf = Mem.alloc(sizeof(Int32))
Mem.upload!(counter_buf, [Int32(0)])
counter_pointer = Base.unsafe_convert(CuPtr{Int32}, counter_buf)

lock_buf = Mem.alloc(sizeof(CUDAnative.MutexState))
Mem.upload!(lock_buf, [CUDAnative.MutexState(0)])
lock_pointer = Base.unsafe_convert(CuPtr{CUDAnative.MutexState}, lock_buf)

# @device_code_warntype increment_counter(counter_pointer, lock_pointer)

# Run the kernel.
@cuda threads=thread_count increment_counter(counter_pointer, lock_pointer)

# Check that the counter's final value equals the number
# of threads.
@test Mem.download(Int32, counter_buf) == [Int32(total_count)]
