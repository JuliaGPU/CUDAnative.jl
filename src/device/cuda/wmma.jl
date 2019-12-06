################################################################################
# CONSTANTS
################################################################################

# Maps PTX types to Julia array types
map_ptx_to_jl_array = Dict(
                           "f16" => Float16,
                           "f32" => Float32
                          )

# Maps PTX types to Julia fragment types
map_ptx_to_jl_frag = Dict(
                          "f16" => NTuple{2, VecElement{Float16}},
                          "f32" => Float32
                         )

# Maps matrix & PTX types to fragment sizes
map_frag_sizes = Dict(
                      "a.f16" => 8,
                      "b.f16" => 8,
                      "c.f16" => 4,
                      "c.f32" => 8,
                      "d.f16" => 4,
                      "d.f32" => 8
                     )

# Maps PTX AS to Int
map_ptx_as_to_int = Dict(
                         "" => 0,
                         "shared" => 3,
                         "global" => 1
                        )

################################################################################
# HELPER FUNCTIONS
################################################################################

function join_nonempty(args...)
    delim = args[end]
    arr = [args[1:end-1]...]

    return join(arr[arr .!= ""], delim)
end

# Returns (Julia array type, Julia fragment type, fragment size)
get_frag_info(matrix, ptx_el_type) = (
        map_ptx_to_jl_array[ptx_el_type],
        map_ptx_to_jl_frag[ptx_el_type],
        map_frag_sizes["$matrix.$ptx_el_type"]
        )

get_addrspace_info(addr_space) = map_ptx_as_to_int[addr_space]

################################################################################
# LOW LEVEL API
################################################################################

# -----------
# Matrix load
# -----------

for mat in ["a", "b", "c"],
    layout in ["col", "row"],
    shape in ["m16n16k16"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"],
    elem_type in ["f16", "f32"]

    # TODO: Non-stride versions?

    # Float32 is only supported for C
    if (elem_type == "f32") && (mat != "c")
        continue
    end

    addr_space_int = get_addrspace_info(addr_space)

    # Name of the Julia wrapper function
    func_name = Symbol(join_nonempty("llvm", "wmma", "load", mat, layout, shape, addr_space, stride, elem_type, "_"))

    # Name of the LLVM intrinsic
    llvm_intr = "llvm.nvvm.wmma.$shape.load.$mat.$layout.stride.$elem_type.p$(addr_space_int)i8"

    # Determine types + size for this (matrix, elem_type) combination
    arr_ty, frag_ty, sz = get_frag_info(mat, elem_type)

    ccall_name = "extern $llvm_intr"

    @eval $func_name(src_addr, stride) = ccall($ccall_name, llvmcall, NTuple{$sz, $frag_ty}, (Ref{$arr_ty}, Int32), src_addr, stride)
    @eval export $func_name
end

# ------------
# Matrix store
# ------------

for mat in ["d"],
    layout in ["col", "row"],
    shape in ["m16n16k16"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"],
    elem_type in ["f16", "f32"]

    # TODO: Non-stride versions?

    addr_space_int = get_addrspace_info(addr_space)

    # Name of the Julia wrapper function
    func_name = Symbol(join_nonempty("llvm", "wmma", "store", mat, layout, shape, addr_space, stride, elem_type, "_"))

    # Name of the LLVM intrinsic
    llvm_intr = "llvm.nvvm.wmma.$shape.store.$mat.$layout.stride.$elem_type.p$(addr_space_int)i8"

    # Determine types + size for this (matrix, elem_type) combination
    arr_ty, frag_ty, sz = get_frag_info(mat, elem_type)

    ccall_name = "extern $llvm_intr"
    frag_types = ntuple(i -> frag_ty, sz)
    frag_vars = ntuple(i -> :(data[$i]), sz)

    @eval $func_name(dst_addr, data, stride) = ccall($ccall_name, llvmcall, Nothing, (Ref{$arr_ty}, $(frag_types...), Int32), dst_addr, $(frag_vars...), stride)
    @eval export $func_name
end

# --------------------------
# Matrix multiply accumulate
# --------------------------

for a_layout in ["col", "row"],
    b_layout in ["col", "row"],
    shape in ["m16n16k16"],
    d_elem_type in ["f16", "f32"],
    c_elem_type in ["f16", "f32"],
    b_elem_type in ["f16"],
    a_elem_type in ["f16"]

    # Name of the Julia wrapper function
    func_name = Symbol(join_nonempty("llvm", "wmma", "mma", a_layout, b_layout, shape, d_elem_type, c_elem_type, "_"))

    # Name of the LLVM intrinsic
    llvm_intr = "llvm.nvvm.wmma.$shape.mma.$a_layout.$b_layout.$d_elem_type.$c_elem_type"

    # Determine types + size for the (matrix, elem_type) combinations for matrix A, B, C and D
    a_arr_ty, a_frag_ty, a_sz = get_frag_info("a", a_elem_type)
    b_arr_ty, b_frag_ty, b_sz = get_frag_info("b", b_elem_type)
    c_arr_ty, c_frag_ty, c_sz = get_frag_info("c", c_elem_type)
    d_arr_ty, d_frag_ty, d_sz = get_frag_info("d", d_elem_type)

    ccall_name = "extern $llvm_intr"

    a_types = ntuple(i -> a_frag_ty, a_sz)
    b_types = ntuple(i -> b_frag_ty, b_sz)
    c_types = ntuple(i -> c_frag_ty, c_sz)

    a_vars = ntuple(i -> :(a[$i]), a_sz)
    b_vars = ntuple(i -> :(b[$i]), b_sz)
    c_vars = ntuple(i -> :(c[$i]), c_sz)

    @eval $func_name(a, b, c) = ccall($ccall_name, llvmcall, NTuple{$d_sz, $d_frag_ty}, ($(a_types...), $(b_types...), $(c_types...)), $(a_vars...), $(b_vars...), $(c_vars...))
    @eval export $func_name
end

################################################################################
# FLATTENING/UNFLATTENING LOGIC
################################################################################

# Base case (Float16, Float32, ...)
flatten_recurse(typ, e) = [:($e)]
unflatten_recurse(typ, e, idx) = :($e[$idx]), idx + 1

# VecElements
flatten_recurse(typ::Type{VecElement{T}}, e) where T = [:($e.value)]
unflatten_recurse(typ::Type{VecElement{T}}, e, idx) where T = :(VecElement{$T}($e[$idx])), idx + 1

# NTuples
function flatten_recurse(typ::Type{NTuple{N, T}}, e) where {N, T}
    ret = Expr[]

    for (i, eltyp) in enumerate(typ.types)
        append!(ret, flatten_recurse(eltyp, :($e[$i])))
    end

    return ret
end

function unflatten_recurse(typ::Type{NTuple{N, T}}, e, idx) where {N, T}
    ret = Expr(:tuple)

    for (i, eltyp) in enumerate(typ.types)
        arg, idx = unflatten_recurse(eltyp, e, idx)
        push!(ret.args, arg)
    end

    return ret, idx
end

@generated flatten(x::typ) where typ = Expr(:tuple, flatten_recurse(typ, :x)...)
@generated unflatten(::Type{typ}, x) where typ = unflatten_recurse(typ, :x, 1)[1]

################################################################################
# HIGH LEVEL (CUDA-STYLE API)
################################################################################

# -------------
# WMMA fragment
# -------------

export WMMAFragmentLayout, WMMARowMajor, WMMAColMajor, WMMAUnspecified

"""
    WMMAFragmentLayout

Abstract type that specifies the storage layout of a matrix.

Possible values are [`WMMARowMajor`](@ref), [`WMMAColMajor`](@ref) and [`WMMAUnspecified`](@ref).
"""
abstract type WMMAFragmentLayout end

"""
    WMMARowMajor

Type that represents a matrix stored in row major (C style) order.
"""
struct WMMARowMajor <: WMMAFragmentLayout end

"""
    WMMAColMajor

Type that represents a matrix stored in column major (Julia style) order.
"""
struct WMMAColMajor <: WMMAFragmentLayout end

"""
    WMMAUnspecified

Type that represents a matrix stored in an unspecified order.

!!! warning

    This storage format is not valid for all WMMA operations!
"""
struct WMMAUnspecified <: WMMAFragmentLayout end


export WMMAMatrixA, WMMAMatrixB, WMMAAccumulator

abstract type WMMAFragmentUse end
struct WMMAMatrixA <: WMMAFragmentUse end
struct WMMAMatrixB <: WMMAFragmentUse end
struct WMMAAccumulator <: WMMAFragmentUse end


export WMMAFragment

"""
    WMMAFragment

Type that represents per-thread intermediate results of WMMA operations.

You can access individual elements using the `x` member, but beware that the exact ordering of elements is unspecified.
"""
struct WMMAFragment{M, N, K, FS, T, L <: WMMAFragmentLayout, U <: WMMAFragmentUse}
    x::NTuple{FS, T}
end

# ------------------
# WMMA configuration
# ------------------

export WMMAConfig

"""
    WMMAConfig{M, N, K, d_type}

Type that contains all information for WMMA operations that cannot be inferred from the argument's types.

WMMA instructions calculate the matrix multiply-accumulate operation ``D = A \\cdot B + C``, where ``A`` is a ``M \\times K`` matrix,
``B`` a ``K \\times N`` matrix, and ``C`` and ``D`` are ``M \\times N`` matrices.

`d_type` refers to the type of the elements of matrix ``D``, and can be either `Float16` or `Float32`.

All WMMA operations take a `WMMAConfig` as their final argument.

# Examples
```jldoctest
julia> config = WMMAConfig{16, 16, 16, Float32}
WMMAConfig{16,16,16,Float32}
```
"""
struct WMMAConfig{M, N, K, d_type} end

# ---------
# Constants
# ---------

# Maps Julia array types to string
map_jl_array_to_str = Dict(val => key for (key, val) in map_ptx_to_jl_array)

# Maps CUDAnative.AS types to string
map_as_ty_to_str = Dict(
                        AS.Generic => "",
                        AS.Shared => "shared",
                        AS.Global => "global"
                       )

# Maps layout types to string
map_layout_ty_to_str = Dict(
                            WMMARowMajor => "row",
                            WMMAColMajor => "col"
                           )

# Maps matrix & type to number of elements (size after flattening)
map_num_elems = Dict(
                     ("a", Float16) => 16,
                     ("b", Float16) => 16,
                     ("c", Float16) => 8,
                     ("c", Float32) => 8,
                     ("d", Float16) => 8,
                     ("d", Float32) => 8
                    )

# Maps matrix to its use
map_matrix_to_use = Dict(
                      "a" => WMMAMatrixA,
                      "b" => WMMAMatrixB,
                      "c" => WMMAAccumulator,
                      "d" => WMMAAccumulator
                        )

# ----------------
# Helper functions
# ----------------

function get_hl_as_info(AS)
    try
        return map_as_ty_to_str[AS]
    catch
        error("Invalid address space for WMMA: $AS")
    end
end

function get_hl_layout(L)
    try
        return map_layout_ty_to_str[L]
    catch
        error("Invalid layout for WMMA: $L")
    end
end

function get_hl_shape(M, N, K)
    if (M, N, K) != (16, 16, 16)
        error("Invalid shape for WMMA: (M, N, K) = ($M, $N, $K)")
    end

    return "m$(M)n$(N)k$(K)"
end

get_hl_mat_use(mat) = map_matrix_to_use[mat]

function get_hl_frag_info(matrix, T)
    ptx_ty = nothing

    try
        ptx_ty = map_jl_array_to_str[T]
    catch
        error("Invalid element type for WMMA: $T")
    end

    try
        return (map_num_elems[(matrix, T)],
                map_frag_sizes["$matrix.$ptx_ty"],
                map_ptx_to_jl_frag[ptx_ty],
                ptx_ty)
    catch
        error("Invalid type $T for matrix $matrix")
    end
end

# ---------
# WMMA load
# ---------

export wmma_load_a, wmma_load_b, wmma_load_c

"""
    wmma_load_a(addr, stride, layout, config)
    wmma_load_b(addr, stride, layout, config)
    wmma_load_c(addr, stride, layout, config)

Load the matrix `a`, `b` or `c` from the memory location indicated by `addr`, and return the resulting [`WMMAFragment`](@ref).

# Arguments
- `addr`: The address to load the matrix from.
- `stride`: The leading dimension of the matrix pointed to by `addr`, specified in number of elements.
- `layout`: The storage layout of the matrix. Possible values are [`WMMARowMajor`](@ref) and [`WMMAColMajor`](@ref).
- `config`: The WMMA configuration that should be used for loading this matrix. See [`WMMAConfig`](@ref).

See also: [`WMMAFragment`](@ref), [`WMMAFragmentLayout`](@ref), [`WMMAConfig`](@ref)

!!! warning

    All threads in a warp **MUST** execute the load operation in lockstep, and have to use exactly the same arguments.
    Failure to do so will result in undefined behaviour.
"""
wmma_load_a, wmma_load_b, wmma_load_c

for mat in ["a", "b", "c"]
    func_name = Symbol("wmma_load_$mat")

    @eval @generated function $func_name(addr::DevicePtr{T, AS},
                                         stride::Number,
                                         layout::Type{L},
                                         config::Type{WMMAConfig{M, N, K, D_TYPE}}) where {T, AS, L, M, N, K, D_TYPE}

        as_str                 = get_hl_as_info(AS)
        layout                 = get_hl_layout(L)
        shape                  = get_hl_shape(M, N, K)
        num_els, _, _, arr_str = get_hl_frag_info($mat, T)
        U                      = get_hl_mat_use($mat)
        L_ret                  = ($mat == "c") ? WMMAUnspecified : L

        # Name of the Julia wrapper
        wrapper = Symbol(join_nonempty("llvm", "wmma", "load", $mat, layout, shape, as_str, "stride", arr_str, "_"))

        return quote
            x = flatten($wrapper(addr, stride))
            return WMMAFragment{$M, $N, $K, $num_els, $T, $L_ret, $U}(x)
        end
    end
end


# ------------------------
# WMMA multiply-accumulate
# ------------------------

export wmma_mma

"""
    wmma_mma(a, b, c, conf)

Perform the matrix multiply-accumulate operation ``D = A \\cdot B + C``.

# Arguments

- `a`: The [`WMMAFragment`](@ref) corresponding to the matrix ``A``.
- `b`: The [`WMMAFragment`](@ref) corresponding to the matrix ``B``.
- `c`: The [`WMMAFragment`](@ref) corresponding to the matrix ``C``.
- `conf`: The [`WMMAConfig`](@ref) that should be used in this WMMA operation.

!!! warning

    All threads in a warp **MUST** execute the `mma` operation in lockstep, and have to use exactly the same arguments.
    Failure to do so will result in undefined behaviour.
"""
wmma_mma

@generated function wmma_mma(a::WMMAFragment{M, N, K, A_SZ, A_T, A_L, WMMAMatrixA},
                             b::WMMAFragment{M, N, K, B_SZ, B_T, B_L, WMMAMatrixB},
                             c::WMMAFragment{M, N, K, C_SZ, C_T, WMMAUnspecified, WMMAAccumulator},
                             config::Type{WMMAConfig{M, N, K, D_T}}) where {M, N, K, A_SZ, A_T, A_L, B_SZ, B_T, B_L, C_SZ, C_T, D_T}

    _, a_frag_sz, a_frag_ty, _         = get_hl_frag_info("a", A_T)
    _, b_frag_sz, b_frag_ty, _         = get_hl_frag_info("b", B_T)
    _, c_frag_sz, c_frag_ty, c_arr_str = get_hl_frag_info("c", C_T)
    d_num_els, _, _, d_arr_str         = get_hl_frag_info("d", D_T)

    a_layout = get_hl_layout(A_L)
    b_layout = get_hl_layout(B_L)
    shape = get_hl_shape(M, N, K)

    # Name of the Julia wrapper
    wrapper = Symbol(join_nonempty("llvm", "wmma", "mma", a_layout, b_layout, shape, d_arr_str, c_arr_str, "_"))

    return quote
        a_unfl = unflatten(NTuple{$a_frag_sz, $a_frag_ty}, a.x)
        b_unfl = unflatten(NTuple{$b_frag_sz, $b_frag_ty}, b.x)
        c_unfl = unflatten(NTuple{$c_frag_sz, $c_frag_ty}, c.x)

        x = flatten($wrapper(a_unfl, b_unfl, c_unfl))
        return WMMAFragment{$M, $N, $K, $d_num_els, $D_T, WMMAUnspecified, WMMAAccumulator}(x)
    end
end


# ----------
# WMMA store
# ----------

export wmma_store_d

"""
    wmma_store_d(addr, d, stride, layout, config)

Store the result matrix `d` to the memory location indicated by `addr`.

# Arguments
- `addr`: The address to store the matrix to.
- `d`: The [`WMMAFragment`](@ref) corresponding to the `d` matrix.
- `stride`: The leading dimension of the matrix pointed to by `addr`, specified in number of elements.
- `layout`: The storage layout of the matrix. Possible values are [`WMMARowMajor`](@ref) and [`WMMAColMajor`](@ref).
- `config`: The WMMA configuration that should be used for storing this matrix. See [`WMMAConfig`](@ref).

See also: [`WMMAFragment`](@ref), [`WMMAFragmentLayout`](@ref), [`WMMAConfig`](@ref)

!!! warning

    All threads in a warp **MUST** execute the `store` operation in lockstep, and have to use exactly the same arguments.
    Failure to do so will result in undefined behaviour.
"""
wmma_store_d

@generated function wmma_store_d(addr::DevicePtr{T, AS},
                                 d::WMMAFragment{M, N, K, D_SZ, T, WMMAUnspecified, WMMAAccumulator},
                                 stride::Number,
                                 layout::Type{L},
                                 config::Type{WMMAConfig{M, N, K, T}}) where {T, AS, M, N, K, D_SZ, L}

    as_str                             = get_hl_as_info(AS)
    layout                             = get_hl_layout(L)
    shape                              = get_hl_shape(M, N, K)
    num_els, frag_sz, frag_ty, arr_str = get_hl_frag_info("d", T)

    # Name of the Julia wrapper
    wrapper = Symbol(join_nonempty("llvm", "wmma", "store", "d", layout, shape, as_str, "stride", arr_str, "_"))

    return quote
        d_unfl = unflatten(NTuple{$frag_sz, $frag_ty}, d.x)
        $wrapper(addr, d_unfl, stride)
        return nothing
    end
end


# ------------------
# WMMA fill fragment
# ------------------

export wmma_fill_c

"""
    wmma_fill_c(value, config)

Return a [`WMMAFragment`](@ref) filled with the value `value`.

This operation is useful if you want to implement a matrix multiplication (and thus want to set ``C = O``).

# Arguments
- `value`: The value used to fill the fragment. Can be a `Float16` or `Float32`.
- `config`: The WMMA configuration that should be used for this WMMA operation. See [`WMMAConfig`](@ref).
"""
wmma_fill_c

@generated function wmma_fill_c(value::T,
                                config::Type{WMMAConfig{M, N, K, D_TYPE}}) where {T, M, N, K, D_TYPE}

    # We can't use closures in @generated functions, so we'll have to do it this way instead of
    # ntuple(i -> val, $num_els)
    num_els, _, _ = get_hl_frag_info("c", T)

    args = [:value for i=1:num_els]
    expr = :(tuple($(args...)))

    return quote
        return WMMAFragment{$M, $N, $K, $num_els, $T, WMMAUnspecified, WMMAAccumulator}($expr)
    end
end

################################################################################
# BROADCASTING OVER WMMA FRAGMENTS
################################################################################

# Based on broadcasting implementation of Tuples in
# https://github.com/JuliaLang/julia/blob/master/base/broadcast.jl


# Custom broadcast style for WMMAFragments
struct WMMAFragmentBroadcastStyle <: Broadcast.BroadcastStyle end

# Use this broadcasting style for WMMAFragments
Base.BroadcastStyle(::Type{<:WMMAFragment}) = WMMAFragmentBroadcastStyle()

# Broadcast style precedence rules
# If we broadcast a fragment with a scalar, we want the WMMAFragment style to take precedence
Base.BroadcastStyle(s::WMMAFragmentBroadcastStyle, t::Broadcast.DefaultArrayStyle{0}) = s

# We don't want to convert fragments before broadcasting
Base.broadcastable(frag::WMMAFragment) = frag

# Needed for broadcast machinery
Base.axes(frag::WMMAFragment) = axes(frag.x)

# Helper functions to get element at specified index
@inline get_index(x, i) = x                           # scalar
@inline get_index(frag::WMMAFragment, i) = frag.x[i] # WMMAFragment

# Helper functions to get first fragment in broadcast call
@inline find_first_fragment(args::Tuple) = find_first_fragment(args[1], Base.tail(args))
@inline find_first_fragment(a::WMMAFragment, tail) = a
@inline find_first_fragment(::Any, tail) = find_first_fragment(tail)

# Custom broadcast implementation that returns a WMMAFragment
@inline function Base.copy(bc::Broadcast.Broadcasted{WMMAFragmentBroadcastStyle})
    dim = Broadcast.combine_axes(bc.args...)

    if length(dim) != 1
        throw(DimensionMismatch("WMMA fragment broadcast only supports one dimension!"))
    end

    N = length(dim[1])

    tuple = ntuple(i -> bc.f(map(arg -> get_index(arg, i), bc.args)...), Val(N))

    frag_ty = typeof(find_first_fragment(bc.args))
    return frag_ty(tuple)
end
