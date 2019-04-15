module ArrayReduction

using CUDAdrv, CUDAnative

# This benchmark approximates pi by naively constructing an array comprehension
# for the Madhava–Leibniz series and computing its sum. It does this a few times
# to achieve a respectable run time.

const thread_count = 256
const series_length = 200
const runs = 20

function iterative_sum(elements::Array{T})::T where T
    result = zero(T)
    for i in elements
        result += i
    end
    return result
end

function kernel(destination)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    unsafe_store!(destination, 0.0, i)
    for _ in 1:runs
        series = [CUDAnative.pow(-1 / 3.0, Float64(k)) / (2.0 * k + 1.0) for k in 0:series_length]
        unsafe_store!(destination, unsafe_load(destination, i) + CUDAnative.sqrt(12.0) * iterative_sum(series), i)
    end
    return
end

end

function array_reduction_benchmark()
    destination_array = Mem.alloc(Float64, ArrayReduction.thread_count)
    destination_pointer = Base.unsafe_convert(CuPtr{Float64}, destination_array)

    # Run the kernel.
    @cuda_sync threads=ArrayReduction.thread_count ArrayReduction.kernel(destination_pointer)

    @test Mem.download(Float64, destination_array, ArrayReduction.thread_count) ≈ ArrayReduction.runs .* fill(Float64(pi), ArrayReduction.thread_count)
end

@cuda_benchmark "array reduction" array_reduction_benchmark()
