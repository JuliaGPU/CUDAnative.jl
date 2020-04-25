export Operator
module Operator

using CUDAnative
using CUDAnative.MatMul
using CUDAnative.Tiling

# -------------------------------------
# Default definition for padded layouts
# -------------------------------------

for f in (:fragtype_a, :fragtype_b, :fragtype_accum, :load_a, :load_b, :load_c, :store_d)
    @eval @inline $f(op, ::Type{Layout.Padded{L, P}}, args...) where {L, P} = $f(op, L, args...)
end

# ----
# WMMA
# ----

struct WMMAOp{M, N, K} end

@inline shape(::Type{WMMAOp{M, N, K}}) where {M, N, K} = (M = M, N = N, K = K)

@inline fragtype_a(::Type{WMMAOp{16, 16, 16}}, ::Type{Layout.AlignedColMajor{Float16}}) = WMMA.Fragment{16, 16, 16, 16, Float16, WMMA.ColMajor, WMMA.MatrixA}
@inline fragtype_b(::Type{WMMAOp{16, 16, 16}}, ::Type{Layout.AlignedColMajor{Float16}}) = WMMA.Fragment{16, 16, 16, 16, Float16, WMMA.ColMajor, WMMA.MatrixB}
@inline fragtype_accum(::Type{WMMAOp{16, 16, 16}}, ::Type{Layout.AlignedColMajor{Float32}}) = WMMA.Fragment{16, 16, 16, 8, Float32, WMMA.Unspecified, WMMA.Accumulator}

function load_a(::Type{WMMAOp{M, N, K}}, ::Type{Layout.AlignedColMajor{Float16}}, workspace, tile::Tile) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ind = Tuple(tile.index) .+ 1
    @inbounds linear_index = LinearIndices(size(workspace))[ind...]
    ptr = pointer(workspace, linear_index)
    return WMMA.load_a(ptr, size(workspace, 1), WMMA.ColMajor, conf)
end

function load_b(::Type{WMMAOp{M, N, K}}, ::Type{Layout.AlignedColMajor{Float16}}, workspace, tile::Tile) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ind = Tuple(tile.index) .+ 1
    @inbounds linear_index = LinearIndices(size(workspace))[ind...]
    ptr = pointer(workspace, linear_index)
    return WMMA.load_b(ptr, size(workspace, 1), WMMA.ColMajor, conf)
end

function load_c(::Type{WMMAOp{M, N, K}}, ::Type{Layout.AlignedColMajor{Float32}}, workspace, tile::Tile) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ind = Tuple(tile.index) .+ 1
    @inbounds linear_index = LinearIndices(size(workspace))[ind...]
    ptr = pointer(workspace, linear_index)
    return WMMA.load_c(ptr, size(workspace, 1), WMMA.ColMajor, conf)
end

function store_d(::Type{WMMAOp{M, N, K}}, ::Type{Layout.AlignedColMajor{Float32}}, workspace, frag, tile::Tile) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ind = Tuple(tile.index) .+ 1
    @inbounds linear_index = LinearIndices(size(workspace))[ind...]
    ptr = pointer(workspace, linear_index)
    WMMA.store_d(ptr, frag, size(workspace, 1), WMMA.ColMajor, conf)
end

function mma(::Type{WMMAOp{M, N, K}}, a_frag, b_frag, c_frag) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    return WMMA.mma(a_frag, b_frag, c_frag, conf)
end

end
