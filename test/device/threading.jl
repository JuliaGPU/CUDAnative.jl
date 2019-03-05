@testset "threading" begin

############################################################################################

@testset "reader-writer lock" begin

@testset "writers only" begin

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

    # Run the kernel.
    @cuda threads=thread_count increment_counter(counter_pointer, lock_pointer)

    # Check that the counter's final value equals the number
    # of threads.
    @test Mem.download(Int32, counter_buf) == [Int32(thread_count)]

end

@testset "readers and writers" begin

    thread_count = 128

    # Define a kernel.
    function mutate_counter_maybe(counter::CUDAnative.DevicePtr{Int32}, lock_state::CUDAnative.DevicePtr{CUDAnative.ReaderWriterLockState})
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        lock = ReaderWriterLock(lock_state)
        # Read the previous counter and update the current counter.
        # Do this many times.
        if i % 16 == 0
            # Some threads get to atomically increment the counter.
            writer_locked(lock) do
                unsafe_store!(counter, unsafe_load(counter) + 1)
            end
        else
            # All the other threads acquire the lock in reader mode
            # and check that the counter's value doesn't change.
            reader_locked(lock) do
                counter_ptr = convert(Ptr{Int32}, convert(Csize_t, counter))
                counter_val = CUDAnative.volatile_load(counter_ptr)
                j = 0
                while j < 10
                    if CUDAnative.volatile_load(counter_ptr) != counter_val
                        throw(ErrorException("oh no"))
                    end
                    j += 1
                end
            end
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

    # Run the kernel.
    @cuda threads=thread_count mutate_counter_maybe(counter_pointer, lock_pointer)

    # Check that the counter's final value equals the number
    # of threads.
    @test Mem.download(Int32, counter_buf) == [Int32(thread_count / 16)]

end

end

end
