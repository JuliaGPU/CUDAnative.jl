@testset "WMMA" begin

################################################################################

    @testset "LLVM intrinsics" begin

        @testset "llvm_wmma_load" begin
            @testset "$(mat)_$(layout)_$(shape)_$(addr_space)_$(elem_type)" for mat in ["a", "b", "c"],
                layout in ["row", "col"],
                shape in ["m16n16k16"],
                addr_space in [""],
                stride in ["stride"],
                elem_type in ["f16", "f32"]

                # TODO: Test address space?

                # Float32 is only supported for C
                if (elem_type == "f32") && (mat != "c")
                    continue
                end

                # Type-dependent variables
                array_ty = elem_type == "f16" ? Float16 : Float32
                expected = elem_type == "f16" ? (VecElement{Float16}(42), VecElement{Float16}(42)) : Float32(42)

                # Get the function name
                func = getfield(Main, Symbol("llvm_wmma_load_$(mat)_$(layout)_$(shape)_stride_$(elem_type)"))

                input      = 42 * ones(array_ty, (16, 16))
                input_dev  = CuArray(input)
                result     = Array{Bool}(undef, 1)
                result_dev = CuArray(result)

                function kernel(input_dev, result_dev)
                    data = func(pointer(input_dev), 16)
                    result_dev[1] = all(val -> val == expected, data)
                    return
                end

                @cuda threads=32 kernel(input_dev, result_dev)
                @test all(Array(result_dev))
            end
        end

        @testset "llvm_wmma_store" begin
            @testset "$(mat)_$(layout)_$(shape)_$(addr_space)_$(elem_type)" for mat in ["d"],
                layout in ["row", "col"],
                shape in ["m16n16k16"],
                addr_space in [""],
                stride in ["stride"],
                elem_type in ["f16", "f32"]

                # TODO: Test address space?

                # Type-dependent variables
                array_ty = elem_type == "f16" ? Float16 : Float32
                data = elem_type == "f16" ?
                    (
                       (VecElement{Float16}(42), VecElement{Float16}(42)),
                       (VecElement{Float16}(42), VecElement{Float16}(42)),
                       (VecElement{Float16}(42), VecElement{Float16}(42)),
                       (VecElement{Float16}(42), VecElement{Float16}(42))
                    ) : (42, 42, 42, 42, 42, 42, 42, 42)

                # Get the function name
                func = getfield(Main, Symbol("llvm_wmma_store_$(mat)_$(layout)_$(shape)_stride_$(elem_type)"))

                output     = Array{array_ty}(undef, (16, 16))
                output_dev = CuArray(output)

                function kernel(output_dev)
                    func(pointer(output_dev), data, 16)
                    return
                end

                @cuda threads=32 kernel(output_dev)
                @test all(Array(output_dev) .== 42.0)
            end
        end

        @testset "llvm_wmma_mma" begin
            @testset "$(a_layout)_$(b_layout)_$(shape)_$(d_elem_type)_$(c_elem_type)" for a_layout in ["row", "col"],
                b_layout in ["row", "col"],
                shape in ["m16n16k16"],
                d_elem_type in ["f16", "f32"],
                c_elem_type in ["f16", "f32"]

                # Type-dependent variables
                d_ty = d_elem_type == "f16" ? Float16 : Float32
                c_ty = c_elem_type == "f16" ? Float16 : Float32

                # Get the function names
                lda_func = getfield(Main, Symbol("llvm_wmma_load_a_$(a_layout)_m16n16k16_stride_f16"))
                ldb_func = getfield(Main, Symbol("llvm_wmma_load_b_$(b_layout)_m16n16k16_stride_f16"))
                ldc_func = getfield(Main, Symbol("llvm_wmma_load_c_col_m16n16k16_stride_$(c_elem_type)"))
                mma_func = getfield(Main, Symbol("llvm_wmma_mma_$(a_layout)_$(b_layout)_m16n16k16_$(d_elem_type)_$(c_elem_type)"))
                std_func = getfield(Main, Symbol("llvm_wmma_store_d_col_m16n16k16_stride_$(d_elem_type)"))

                # Generate input matrices
                a     = rand(Float16, (16, 16))
                a_dev = CuArray(a)
                b     = rand(Float16, (16, 16))
                b_dev = CuArray(b)
                c     = rand(c_ty, (16, 16))
                c_dev = CuArray(c)

                # Reserve space for result
                d     = Array{d_ty}(undef, (16, 16))
                d_dev = CuArray(d)

                # Matrix MAC kernel (D = A * B + C)
                function kernel(a_dev, b_dev, c_dev, d_dev)
                    a_frag = lda_func(pointer(a_dev), 16)
                    b_frag = ldb_func(pointer(b_dev), 16)
                    c_frag = ldc_func(pointer(c_dev), 16)

                    d_frag = mma_func(a_frag, b_frag, c_frag)

                    std_func(pointer(d_dev), d_frag, 16)
                    return
                end

                @cuda threads=32 kernel(a_dev, b_dev, c_dev, d_dev)

                new_a = (a_layout == "col" ? a : transpose(a))
                new_b = (b_layout == "col" ? b : transpose(b))

                @test new_a * new_b + c ≈ Array(d_dev) rtol=0.01
            end
        end
    end

################################################################################

    @testset "CUDA C-style API" begin

        @testset "MAC: A: $a_layout, B: $b_layout, C: $c_layout, D: $d_layout, C type: $c_type, D type: $d_type" for a_layout in [wmma_col_major, wmma_row_major],
            b_layout in [wmma_col_major, wmma_row_major],
            c_layout in [wmma_col_major, wmma_row_major],
            d_layout in [wmma_col_major, wmma_row_major],
            c_type in [Float16, Float32],
            d_type in [Float16, Float32]

            a     = rand(Float16, (16, 16))
            b     = rand(Float16, (16, 16))
            c     = rand(c_type, (16, 16))
            d     = Array{d_type}(undef, (16, 16))

            a_dev = CuArray(a)
            b_dev = CuArray(b)
            c_dev = CuArray(c)
            d_dev = CuArray(d)

            @eval function kernel(a_dev, b_dev, c_dev, d_dev)
                conf = wmma_config{16, 16, 16, $d_type}

                a_frag = wmma_load_a(pointer(a_dev), 16, $a_layout, conf)
                b_frag = wmma_load_b(pointer(b_dev), 16, $b_layout, conf)
                c_frag = wmma_load_c(pointer(c_dev), 16, $c_layout, conf)

                d_frag = wmma_mma(a_frag, b_frag, c_frag, conf)

                wmma_store_d(pointer(d_dev), d_frag, 16, $d_layout, conf)

                return
            end

            @cuda threads=32 kernel(a_dev, b_dev, c_dev, d_dev)
            d = Array(d_dev)

            new_a = (a_layout == wmma_col_major) ? a : transpose(a)
            new_b = (b_layout == wmma_col_major) ? b : transpose(b)
            new_c = (c_layout == wmma_col_major) ? c : transpose(c)
            new_d = (d_layout == wmma_col_major) ? d : transpose(d)

            @test new_a * new_b + new_c ≈ new_d rtol=0.01
        end

    end

################################################################################
end
