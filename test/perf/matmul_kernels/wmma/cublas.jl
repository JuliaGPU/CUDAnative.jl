using CUDAapi
using CUDAdrv
using CUDAnative
using CUDAnative.MatMul
using CuArrays

M = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])
K = parse(Int, ARGS[3])

function benchmark_matmul(a, b, c, d)
    CuArrays.@sync begin
        CUBLAS.cublasSetMathMode(CUBLAS.handle(), CUBLAS.CUBLAS_TENSOR_OP_MATH)
        CUBLAS.cublasGemmEx(CUBLAS.handle(), CUBLAS.CUBLAS_OP_N, CUBLAS.CUBLAS_OP_N, M, N, K, [Float32(1)], a, CUDAapi.R_16F, M, b, CUDAapi.R_16F, K, [Float32(1)], c, CUDAapi.R_32F, M, CUDAapi.R_32F, CUBLAS.CUBLAS_GEMM_DEFAULT)
    end
end

a_h = rand(Float16, (M, K)) / sqrt(Float16(K))
b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
c_h = rand(Float32, (M, N))

a   = CuArray(a_h)
b   = CuArray(b_h)
c   = CuArray(c_h)
d   = similar(c)

# warmup
benchmark_matmul(a, b, c, d)

# profile
for i = 1 : 10
    CUDAdrv.@profile benchmark_matmul(a, b, c, d)
end
