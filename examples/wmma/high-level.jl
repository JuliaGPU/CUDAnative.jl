# Need https://github.com/JuliaLang/julia/pull/33970
if VERSION >= v"1.4.0-DEV.534"

### START
using CUDAnative
using CuArrays
using Test

a     = rand(Float16, (16, 16))
b     = rand(Float16, (16, 16))
c     = rand(Float32, (16, 16))

a_dev = CuArray(a)
b_dev = CuArray(b)
c_dev = CuArray(c)
d_dev = similar(c_dev)

function kernel(a_dev, b_dev, c_dev, d_dev)
    conf = WMMAConfig{16, 16, 16, Float32}

    a_frag = wmma_load_a(pointer(a_dev), 16, WMMAColMajor, conf)
    b_frag = wmma_load_b(pointer(b_dev), 16, WMMAColMajor, conf)
    c_frag = wmma_load_c(pointer(c_dev), 16, WMMAColMajor, conf)

    d_frag = wmma_mma(a_frag, b_frag, c_frag, conf)

    wmma_store_d(pointer(d_dev), d_frag, 16, WMMAColMajor, conf)

    return
end

@cuda threads=32 kernel(a_dev, b_dev, c_dev, d_dev)
d = Array(d_dev)

@test all(isapprox.(a * b + c, d; rtol=0.01))
### END

end
