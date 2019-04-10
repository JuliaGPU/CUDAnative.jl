module StreamQueries

using CUDAnative, CUDAdrv
import ..LinkedList: List, Nil, Cons, foldl, map, max, filter

# This benchmark applies stream operators (map, max,filter) to purely
# functional lists.

const thread_count = 256
const input_size = 100

function kernel(input::CUDAnative.DevicePtr{Float64}, output::CUDAnative.DevicePtr{Float64})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    values = List{Float64}(input, input_size)
    values = map(x -> x * x, values)
    values = filter(x -> x < 10.0 && x >= 0.0, values)
    unsafe_store!(output, max(x -> x, values, 0.0), i)
end

end

function stream_benchmark()
    source_array = Mem.alloc(Float64, StreamQueries.input_size)
    Mem.upload!(source_array, rand(Float64, StreamQueries.input_size))
    destination_array = Mem.alloc(Float64, StreamQueries.thread_count)
    source_pointer = Base.unsafe_convert(CuPtr{Float64}, source_array)
    destination_pointer = Base.unsafe_convert(CuPtr{Float64}, destination_array)
    @cuda_sync threads=StreamQueries.thread_count StreamQueries.kernel(source_pointer, destination_pointer)
end

@cuda_benchmark "stream queries" stream_benchmark()
