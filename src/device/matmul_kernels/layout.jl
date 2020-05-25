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
@inline load(::Type{Padded{L, P}}, workspace, tile::Tile, workspace_size::NamedTuple) where {L, P} = load(L, workspace, tile, pad_logical_coord(Padded{L, P}, workspace_size))
@inline store!(::Type{Padded{L, P}}, workspace, value, tile::Tile, workspace_size::NamedTuple) where {L, P} = store!(L, workspace, value, tile::Tile, pad_logical_coord(Padded{L, P}, workspace_size))

# ---------------
# AlignedColMajor
# ---------------

struct AlignedColMajor{T} <: LayoutBase{T} end

# TODO: cleanup vectorisation
@inline function load(::Type{AlignedColMajor{T}}, workspace, tile::Tile{size}, workspace_size::NamedTuple) where {T, size}
    vec_len = 16 ÷ sizeof(T)
    N = (sizeof(T) * vec_len) ÷ sizeof(Float32)
    res = MArray{Tuple{size[1] ÷ vec_len, size[2]}, NTuple{N, VecElement{Float32}}}(undef)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : vec_len : size[1]
            t = translate(tile, (i - 1, j - 1))

            linear_base = linearise(t.base, workspace_size)
            linear_offset = linearise(t.offset, workspace_size)

            @inbounds res[i, j] = vloada(Vec{vec_len, T}, pointer(workspace, linear_base), linear_offset)
        end
    end

    return res
end

@inline function store!(::Type{AlignedColMajor{T}}, workspace, value, tile::Tile{size}, workspace_size::NamedTuple) where {T, size}
    vec_len = 16 ÷ sizeof(T)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : vec_len : size[1]
            t = translate(tile, (i - 1, j - 1))

            linear_base = linearise(t.base, workspace_size)
            linear_offset = linearise(t.offset, workspace_size)

            vstorea!(Vec{vec_len, T}, pointer(workspace, linear_base), value[i, j], linear_offset)
        end
    end
end

end
