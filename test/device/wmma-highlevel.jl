@testset "WMMA" begin
    @testset "CUDA C-style API" begin

        @testset "One specific case" begin
            a     = rand(Float16, (16, 16))
            b     = rand(Float16, (16, 16))
            c     = rand(Float16, (16, 16))
            d     = Array{Float16}(undef, (16, 16))

            a_dev = CuArray(a)
            b_dev = CuArray(b)
            c_dev = CuArray(c)
            d_dev = CuArray(d)

            function kernel(a_dev, b_dev, c_dev, d_dev)
                conf = wmma_config{16, 16, 16}

                a_frag = wmma_load_a(pointer(a_dev), 16, wmma_col_major, conf)
                b_frag = wmma_load_b(pointer(b_dev), 16, wmma_col_major, conf)
                c_frag = wmma_load_c(pointer(c_dev), 16, wmma_col_major, conf)

                d_frag = wmma_mma(a_frag, b_frag, c_frag)

                wmma_store_d(pointer(d_dev), d_frag, 16, wmma_col_major, conf)

                return
            end

            @cuda threads=32 kernel(a_dev, b_dev, c_dev, d_dev)
            d = Array(d_dev)
            @test a * b + c ≈ d rtol=0.01
        end

    end
end
