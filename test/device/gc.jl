@testset "gc" begin

############################################################################################

dummy() = return

dummy_handler(kernel) = return

@testset "@cuda_gc" begin

@testset "allocate and collect" begin
    # This test allocates many very small and very large objects. Both the small
    # and large objects become garbage eventually, but small objects need to
    # outlive the large objects (and not be collected erroneously) for the test
    # to pass. So essentially this test tackles three things:
    #
    #   1. Allocation works.
    #   2. Collection works.
    #   3. Collection isn't gung-ho to the point of incorrectness.
    #

    mutable struct TempStruct
        data::Float32
    end

    @noinline function escape(val)
        Base.pointer_from_objref(val)
    end

    # Define a kernel that copies values using a temporary struct.
    function kernel(a::CUDAnative.DevicePtr{Float32}, b::CUDAnative.DevicePtr{Float32})
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

        for j in 1:2
            # Allocate a mutable struct and make sure it ends up on the GC heap.
            temp = TempStruct(unsafe_load(a, i))
            escape(temp)

            # Allocate a large garbage buffer to force collections.
            gc_malloc(Csize_t(256 * 1024))

            # Use the mutable struct. If its memory has been reclaimed (by accident)
            # then we expect the test at the end of this file to fail.
            unsafe_store!(b, temp.data, i)
        end

        return
    end

    thread_count = 64

    # Allocate two arrays.
    source_array = Mem.alloc(Float32, thread_count)
    destination_array = Mem.alloc(Float32, thread_count)
    source_pointer = Base.unsafe_convert(CuPtr{Float32}, source_array)
    destination_pointer = Base.unsafe_convert(CuPtr{Float32}, destination_array)

    # Fill the source and destination arrays.
    Mem.upload!(source_array, fill(42.f0, thread_count))
    Mem.upload!(destination_array, zeros(Float32, thread_count))

    # Run the kernel.
    @cuda_gc threads=thread_count kernel(source_pointer, destination_pointer)

    @test Mem.download(Float32, destination_array, thread_count) == fill(42.f0, thread_count)
end

end

end
