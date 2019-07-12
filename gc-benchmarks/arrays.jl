module Arrays

using CUDAdrv, CUDAnative
import ..CUDArandom: LinearCongruentialGenerator, next

# This benchmark allocates a hierarchy of fairly modest Julia arrays.
# Some arrays remain alive, others become unreachable. This benchmark
# seeks to ascertain the performance of the allocator and garbage collector.

const thread_count = 64
const insertion_count = 80

function insert(target::Array{Any, 1}, generator::LinearCongruentialGenerator)
    while true
        index = next(generator, 1, length(target))
        elem = target[index]
        if isa(elem, Array{Any, 1})
            if length(elem) > 0
                if next(generator, 0, 2) == 0
                    target = elem
                    continue
                end
            end
        end

        target[index] = Any[Any[] for _ in 1:5]
        return
    end
end

function kernel()
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    generator = LinearCongruentialGenerator(i)
    toplevel = Any[Any[] for _ in 1:10]
    for i in 1:insertion_count
        insert(toplevel, generator)
    end
    return
end

end

function arrays_benchmark()
    # Run the kernel.
    @cuda_sync threads=Arrays.thread_count Arrays.kernel()
end

@cuda_benchmark "arrays" arrays_benchmark()
