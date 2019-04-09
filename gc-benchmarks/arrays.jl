module Arrays

using CUDAdrv, CUDAnative, StaticArrays

# This benchmark allocates a variety of differently-sized arrays.
# The point of this benchmark is to ascertain how well the GC handles
# many differently-sized objects.

const thread_count = 64

@noinline function escape(value)
    Base.pointer_from_objref(value)
    value
end

macro new_array(T, size)
    quote
        escape(zeros(MArray{Tuple{$size}, $T}))
    end
end

function kernel()
    for i in 1:2
        for j in 1:2
            for k in 1:2
                for l in 1:2
                    @new_array(Int64, 4)
                    @new_array(Int64, 8)
                    @new_array(Int64, 16)
                end
                @new_array(Int64, 32)
                @new_array(Int64, 64)
                @new_array(Int64, 128)
            end
            @new_array(Int64, 256)
            @new_array(Int64, 512)
            @new_array(Int64, 1024)
        end
        @new_array(Int64, 2048)
        @new_array(Int64, 4096)
        @new_array(Int64, 8192)
    end
    return
end

end

function arrays_benchmark()
    # Run the kernel.
    @cuda_sync threads=Arrays.thread_count Arrays.kernel()
end

@cuda_benchmark "arrays" arrays_benchmark()
