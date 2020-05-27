using CUDAnative
using CUDAnative.MatMul

################################################################################

@testset "Matmul API" begin
    @testset "WMMA GEMM" begin
        @testset "(M = $M, N = $N, K = $K)" for M in [128, 256, 1024, 2048],
            N in [128, 256, 1024, 2048],
            K in [128, 256, 1024, 2048]

            a_h = rand(Float16, (M, K)) / sqrt(Float16(K))
            b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
            c_h = rand(Float32, (M, N))

            a   = CuArray(a_h)
            b   = CuArray(b_h)
            c   = CuArray(c_h)
            d   = similar(c)

            conf = MatMul.get_config(
                gemm_shape = (M = M, N = N, K = K),
                operator = Operator.WMMAOp{16, 16, 16},
                global_a_layout = Layout.AlignedColMajor{Float16},
                global_c_layout = Layout.AlignedColMajor{Float32}
                                    )

            MatMul.matmul(a, b, c, d, conf)

            @test all(isapprox.(Float32.(a_h) * Float32.(b_h) + c_h, Array(d); rtol = sqrt(eps(Float16))))
        end
    end

    @testset "WMMA GEMM + scaling" begin
        @testset "(M = $M, N = $N, K = $K, alpha = $alpha)" for M in [128, 256],
            N in [128, 256],
            K in [128, 256],
            alpha in [2, 5]

            a_h = rand(Float16, (M, K)) / sqrt(Float16(K))
            b_h = rand(Float16, (K, N)) / sqrt(Float16(K))
            c_h = rand(Float32, (M, N))

            a   = CuArray(a_h)
            b   = CuArray(b_h)
            c   = CuArray(c_h)
            d   = similar(c)

            conf = MatMul.get_config(
                gemm_shape = (M = M, N = N, K = K),
                operator = Operator.WMMAOp{16, 16, 16},
                global_a_layout = Layout.AlignedColMajor{Float16},
                global_c_layout = Layout.AlignedColMajor{Float32}
                                    )

            MatMul.matmul(a, b, c, d, conf;
                          transform_shared_to_regs_c = Transform.Elementwise(x -> x * alpha))

            @test all(isapprox.(Float32.(a_h) * Float32.(b_h) + alpha * c_h, Array(d); rtol = sqrt(eps(Float16))))
        end
    end

    @testset "WMMA Complex GEMM" begin
        @testset "(M = $M, N = $N, K = $K)" for M in [128, 256, 1024, 2048],
            N in [128, 256, 1024, 2048],
            K in [128, 256, 1024, 2048]

            a_h = rand(Complex{Float16}, (M, K)) / sqrt(Float16(K));
            b_h = rand(Complex{Float16}, (K, N)) / sqrt(Float16(K));
            c_h = rand(Complex{Float32}, (M, N));

            a = CuArray(a_h);
            b = CuArray(b_h);
            c = CuArray(c_h);
            d = similar(c);

            conf = MatMul.get_config(
                    gemm_shape = (M = M, N = N, K = K),
                    operator = Operator.WMMAComplexOp{16, 16, 16},

                    global_a_layout = Layout.InterleavedComplex{Float16},
                    global_b_layout = Layout.InterleavedComplex{Float16},
                    global_c_layout = Layout.InterleavedComplex{Float32},
                    global_d_layout = Layout.InterleavedComplex{Float32},

                    shared_a_layout = Layout.Padded{Layout.SplitComplex{Float16}, 8},
                    shared_b_layout = Layout.Padded{Layout.SplitComplex{Float16}, 8},
                    shared_c_layout = Layout.SplitComplex{Float32},
                    shared_d_layout = Layout.SplitComplex{Float32},

                    warps_per_block = 8,

                    compute_warp = (M = 16, N = 32, K = 16),

                    block_shape = (M = 64, N = 64, K = 32),

                    mem_a_warp = (M = 64, K = 2),
                    mem_b_warp = (K = 32, N = 4),
                    mem_cd_warp = (M = 64, N = 1),

                    mem_a_thread = (M = 4, K = 1),
                    mem_b_thread = (K = 4, N = 1),
                    mem_cd_thread = (M = 2, N = 1)
                )

            MatMul.matmul(a, b, c, d, conf)

            @test all(isapprox.(a_h * b_h + c_h, Array(d); rtol=sqrt(eps(Float16))));
        end
    end
end

################################################################################
