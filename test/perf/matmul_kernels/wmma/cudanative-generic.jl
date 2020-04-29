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

a = CuArray(rand(T, (M, K)));
b = CuArray(rand(T, (K, N)));
c = CuArray(rand(T, (M, N)));

# warmup
GPUArrays.generic_matmatmul!(c, b, a, T(1), T(1))

# profile
for i = 1 : 10
    CUDAdrv.@profile GPUArrays.generic_matmatmul!(c, b, a, T(1), T(1))
end
