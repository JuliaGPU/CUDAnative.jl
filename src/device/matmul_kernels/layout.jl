export Layout
module Layout

using CUDAnative
using CUDAnative.Tiling

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
@inline load(::Type{Padded{L, P}}, workspace, tile::Tile, logical_size::NamedTuple) where {L, P} = load(L, workspace, tile, pad_logical_coord(Padded{L, P}, logical_size))
@inline store!(::Type{Padded{L, P}}, workspace, value, tile::Tile, logical_size::NamedTuple) where {L, P} = store!(L, workspace, value, tile::Tile, pad_logical_coord(Padded{L, P}, logical_size))

# ---------------
# AlignedColMajor
# ---------------

struct AlignedColMajor{T} <: LayoutBase{T} end

@inline function load(::Type{AlignedColMajor{T}}, workspace, tile::Tile, logical_size::NamedTuple) where {T}
    N = 16 ÷ sizeof(T)
    ptr = pointer(workspace, linearise(tile.base, logical_size))
    return vloada(Vec{N, T}, ptr, linearise(tile.offset, logical_size))
end

@inline function store!(::Type{AlignedColMajor{T}}, workspace, value, tile::Tile, logical_size::NamedTuple) where {T}
    N = 16 ÷ sizeof(T)
    ptr = pointer(workspace, linearise(tile.base, logical_size))
    return vstorea!(Vec{N, T}, ptr, value, linearise(tile.offset, logical_size))
end

end
