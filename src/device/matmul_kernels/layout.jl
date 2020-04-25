export Layout
module Layout

using CUDAnative
using CUDAnative.Tiling
using GPUifyLoops
using StaticArrays

# -----------
# Layout base
# -----------

abstract type LayoutBase{T} end

@inline eltype(::Type{<:LayoutBase{T}}) where {T} = T
@inline size(::Type{<:LayoutBase{T}}, logical_size::NamedTuple) where {T} = Tuple(logical_size)

# --------------
# Padded layouts
# --------------

struct Padded{L, P} end

@inline function pad_logical_coord(::Type{Padded{L, P}}, crd::NamedTuple) where {L, P}
    t = Tuple(crd)
    return typeof(crd)((Base.first(t) + P, Base.tail(t)...))
end

@inline eltype(::Type{Padded{L, P}}) where {L, P} = eltype(L)
@inline size(::Type{Padded{L, P}}, logical_size::NamedTuple) where {L, P} = size(L, pad_logical_coord(Padded{L, P}, logical_size))
@inline load(::Type{Padded{L, P}}, workspace, tile::Tile, logical_size::NamedTuple) where {L, P} = load(L, workspace, tile)
@inline store!(::Type{Padded{L, P}}, workspace, value, tile::Tile) where {L, P} = store!(L, workspace, value, tile::Tile)

# ---------------
# AlignedColMajor
# ---------------

struct AlignedColMajor{T} <: LayoutBase{T} end

# TODO: cleanup vectorisation
@inline function load(::Type{AlignedColMajor{T}}, workspace, tile::Tile{size}) where {T, size}
    vec_len = 16 ÷ sizeof(T)
    N = (sizeof(T) * vec_len) ÷ sizeof(Float32)
    res = MArray{Tuple{size[1] ÷ vec_len, size[2]}, NTuple{N, VecElement{Float32}}}(undef)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : vec_len : size[1]
            t = translate(tile, (i - 1, j - 1))
            ind = Tuple(t.index) .+ 1
            @inbounds linear_index = LinearIndices(Base.size(workspace))[ind...]
            @inbounds res[i, j] = vloada(Vec{vec_len, T}, pointer(workspace), linear_index)
        end
    end

    return res
end

@inline function store!(::Type{AlignedColMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    vec_len = 16 ÷ sizeof(T)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : vec_len : size[1]
            t = translate(tile, (i - 1, j - 1))
            ind = Tuple(t.index) .+ 1
            @inbounds linear_index = LinearIndices(Base.size(workspace))[ind...]
            vstorea!(Vec{vec_len, T}, pointer(workspace), value[i, j], linear_index)
        end
    end
end

end
