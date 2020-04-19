export Operator
module Operator

using CUDAnative
using CUDAnative.MatMul
using CUDAnative.Tiling

# -------------------------------------
# Default definition for padded layouts
# -------------------------------------

# Fragment types
for f in (:fragtype_a, :fragtype_b, :fragtype_accum)
    @eval @inline $f(op, ::Type{Layout.Padded{L, P}}, args...) where {L, P} = $f(op, L, args...)
end

# Load fragments
for f in (:load_a, :load_b, :load_c)
    @eval @inline $f(op, ::Type{Layout.Padded{L, P}}, workspace, tile::Tile, logical_size::NamedTuple) where {L, P} = $f(op, L, workspace, tile, Layout.pad_logical_coord(Layout.Padded{L, P}, logical_size))
end

# Store fragments
@inline store_d(op, ::Type{Layout.Padded{L, P}}, workspace, frag, tile::Tile, logical_size::NamedTuple) where {L, P} = store_d(op, L, workspace, frag, tile, Layout.pad_logical_coord(Layout.Padded{L, P}, logical_size))

# ----
# WMMA
# ----

struct WMMAOp{M, N, K} end

@inline shape(::Type{WMMAOp{M, N, K}}) where {M, N, K} = (M = M, N = N, K = K)

@inline fragtype_a(::Type{WMMAOp{16, 16, 16}}, ::Type{Layout.AlignedColMajor{Float16}}) = WMMA.Fragment{16, 16, 16, 16, Float16, WMMA.ColMajor, WMMA.MatrixA}
@inline fragtype_b(::Type{WMMAOp{16, 16, 16}}, ::Type{Layout.AlignedColMajor{Float16}}) = WMMA.Fragment{16, 16, 16, 16, Float16, WMMA.ColMajor, WMMA.MatrixB}
@inline fragtype_accum(::Type{WMMAOp{16, 16, 16}}, ::Type{Layout.AlignedColMajor{Float32}}) = WMMA.Fragment{16, 16, 16, 8, Float32, WMMA.Unspecified, WMMA.Accumulator}

function load_a(::Type{WMMAOp{M, N, K}}, ::Type{Layout.AlignedColMajor{Float16}}, workspace, tile::Tile, logical_size::NamedTuple) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ptr = pointer(workspace, linearise(tile.index, logical_size))
    return WMMA.load_a(ptr, logical_size.M, WMMA.ColMajor, conf)
end

function load_b(::Type{WMMAOp{M, N, K}}, ::Type{Layout.AlignedColMajor{Float16}}, workspace, tile::Tile, logical_size::NamedTuple) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ptr = pointer(workspace, linearise(tile.index, logical_size))
    return WMMA.load_b(ptr, logical_size.K, WMMA.ColMajor, conf)
end

function load_c(::Type{WMMAOp{M, N, K}}, ::Type{Layout.AlignedColMajor{Float32}}, workspace, tile::Tile, logical_size::NamedTuple) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ptr = pointer(workspace, linearise(tile.index, logical_size))
    return WMMA.load_c(ptr, logical_size.M, WMMA.ColMajor, conf)
end

function store_d(::Type{WMMAOp{M, N, K}}, ::Type{Layout.AlignedColMajor{Float32}}, workspace, frag, tile::Tile, logical_size::NamedTuple) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ptr = pointer(workspace, linearise(tile.index, logical_size))
    WMMA.store_d(ptr, frag, logical_size.M, WMMA.ColMajor, conf)
end

function mma(::Type{WMMAOp{M, N, K}}, a_frag, b_frag, c_frag) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    return WMMA.mma(a_frag, b_frag, c_frag, conf)
end

end
