using CUDAnative, CuArrays, GPUArrays, CUDAdrv;

M = parse(Int, ARGS[1]);
N = parse(Int, ARGS[2]);
K = parse(Int, ARGS[3]);

if ARGS[4] == "FP32"
    T = Float32;
elseif ARGS[4] == "FP16"
    T = Float16;
else
    error("Invalid type: $(ARGS[4])");
end

a_h = rand(Complex{T}, (M, K)) / sqrt(T(K));
b_h = rand(Complex{T}, (K, N)) / sqrt(T(K));
c_h = rand(Complex{T}, (M, N));

a = CuArray(a_h);
b = CuArray(b_h);
c = CuArray(c_h);
d = similar(c);

# warmup
GPUArrays.generic_matmatmul!(c, b, a, T(1), T(1))

# profile
for i = 1 : 10
    CUDAdrv.@profile GPUArrays.generic_matmatmul!(c, b, a, T(1), T(1))
end
