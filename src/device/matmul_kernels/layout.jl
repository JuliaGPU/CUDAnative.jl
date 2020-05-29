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
    res = MArray{Tuple{size[1], size[2]}, T}(undef)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : size[1]
            t = translate(tile, (i - 1, j - 1))

            linear_base = linearise(t.base, Base.size(workspace))
            linear_offset = linearise(t.offset, Base.size(workspace))

            @inbounds res[i, j] = workspace[linear_base + linear_offset - 1]
        end
    end

    return res
end

@inline function store!(::Type{AlignedColMajor{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for j = 1 : size[2]
        @unroll for i = 1 : size[1]
            t = translate(tile, (i - 1, j - 1))

            linear_base = linearise(t.base, Base.size(workspace))
            linear_offset = linearise(t.offset, Base.size(workspace))

            @inbounds workspace[linear_base + linear_offset - 1] = value[i,j]
        end
    end
end

# ------------------
# InterleavedComplex
# ------------------

struct InterleavedComplex{T} <: LayoutBase{T} end

@inline function load(::Type{InterleavedComplex{T}}, workspace, tile::Tile{size}) where {T, size}
    res = MArray{Tuple{tile.size[1], tile.size[2]}, Complex{T}}(undef)

    @unroll for j = 1 : tile.size[2]
        @unroll for i = 1 : tile.size[1]
            t = translate(tile, (i - 1, j - 1))

            @inbounds res[i, j] = workspace[t.index[1] + 1, t.index[2] + 1]
        end
    end

    return res
end

@inline function store!(::Type{InterleavedComplex{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for j = 1 : size[2]
        @unroll for i = 1 : size[1]
            t = translate(tile, (i - 1, j - 1))

            @inbounds workspace[t.index[1] + 1, t.index[2] + 1] = value[i, j]
        end
    end
end

# ------------
# SplitComplex
# ------------

struct SplitComplex{T} <: LayoutBase{T} end

@inline function size(::Type{SplitComplex{T}}, logical_size::NamedTuple) where {T}
    t = Tuple(logical_size)
    return (t..., 2)
end

@inline function load(::Type{SplitComplex{T}}, workspace, tile::Tile{size}) where {T, size}
    res = MArray{Tuple{tile.size[1], tile.size[2]}, Complex{T}}(undef)

    @unroll for j = 1 : tile.size[2]
        @unroll for i = 1 : tile.size[1]
            t = translate(tile, (i - 1, j - 1))

            @inbounds res[i,j] = workspace[t.index[1] + 1, t.index[2] + 1, 1] + workspace[t.index[1] + 1, t.index[2] + 1, 2] * im
        end
    end

    return res
end

@inline function store!(::Type{SplitComplex{T}}, workspace, value, tile::Tile{size}) where {T, size}
    @unroll for j = 1 : tile.size[2]
        @unroll for i = 1 : tile.size[1]
            t = translate(tile, (i - 1, j - 1))

            @inbounds workspace[t.index[1] + 1, t.index[2] + 1, 1] = value[i, j].re
            @inbounds workspace[t.index[1] + 1, t.index[2] + 1, 2] = value[i, j].im
        end
    end
end

end
