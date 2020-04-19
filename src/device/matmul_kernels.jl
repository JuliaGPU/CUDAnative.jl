export MatMul
module MatMul

using CUDAdrv
using CUDAnative

include("matmul_kernels/layout.jl")
include("matmul_kernels/operator.jl")
include("matmul_kernels/transform.jl")
include("matmul_kernels/config.jl")
include("matmul_kernels/epilogue.jl")
include("matmul_kernels/kernel.jl")

function matmul(a, b, c, d, conf;
                transform_global_to_shared_a = Transform.Elementwise(),
                transform_global_to_shared_b = Transform.Elementwise(),
                transform_global_to_shared_c = Transform.Elementwise(),
                transform_shared_to_global_d = Transform.Elementwise(),
                transform_shared_to_regs_a = Transform.Elementwise(),
                transform_shared_to_regs_b = Transform.Elementwise(),
                transform_shared_to_regs_c = Transform.Elementwise(),
                transform_regs_to_shared_d = Transform.Elementwise(),
                epilogue = Epilogue.Default())

    args = [a, b, c, d,
            transform_global_to_shared_a, transform_global_to_shared_b, transform_global_to_shared_c, transform_shared_to_global_d,
            transform_shared_to_regs_a, transform_shared_to_regs_b, transform_shared_to_regs_c, transform_regs_to_shared_d,
            epilogue,
            conf]

    GC.@preserve args begin
        kernel_args = cudaconvert.(args)
        kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
        kernel = cufunction(Kernel.matmul_impl, kernel_tt; )
        CUDAdrv.attributes(kernel.fun)[CUDAdrv.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = 64 * 1024
        kernel(kernel_args...; conf.launch_args...)
    end
end

end
