using CUDAdrv, CUDAnative, StaticArrays

# This example allocates an array in a GPU kernel.

const thread_count = 64

@noinline function escape(value)
    Base.pointer_from_objref(value)
    value
end

function kernel()
    array = [1, 2, 3, 4, 5, 6, 7]
    escape(array)
    comp = [i * i for i in array]
    escape(comp)
    return
end

@cuda_gc threads=thread_count kernel()
