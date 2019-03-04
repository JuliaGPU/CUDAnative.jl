@testset "interrupts" begin

############################################################################################

dummy() = return

dummy_handler(kernel) = return

@testset "@cuda_interruptible" begin

@test_throws UndefVarError @cuda_interruptible dummy_handler undefined()
@test_throws MethodError @cuda_interruptible dummy_handler dummy(1)

@testset "compilation params" begin
    @cuda_interruptible dummy_handler dummy()

    @test_throws CuError @cuda_interruptible dummy_handler threads=2 maxthreads=1 dummy()
    @cuda_interruptible dummy_handler threads=2 dummy()
end

@testset "count" begin

    # This test uses interrupts to increment a host counter and then
    # checks that the counter's value equals the number of interrupts.
    # This is a useful thing to check because it verifies that interrupts
    # are neither skipped nor performed twice.
    #
    # We will use a sizeable number of threads (128) to give us a better
    # shot at detecting concurrency errors, if any. The number of skipped
    # interrupts is unlikely to equal the number of additional, unwanted
    # interrupts for this many threads.
    thread_count = 128

    # Define a kernel that makes the host count.
    function increment_counter()
        interrupt()
        return
    end

    # Configure the interrupt to increment a counter.
    global counter = 0
    function handle_interrupt()
        global counter
        counter += 1
    end

    # Run the kernel.
    @cuda_interruptible handle_interrupt threads=thread_count increment_counter()

    # Check that the counter's final value equals the number
    # of threads.
    @test counter == thread_count
end

end

end
