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

export wmma_fragment_layout, wmma_row_major, wmma_col_major, wmma_unspecified

"""
    wmma_fragment_layout

Abstract type that specifies the storage layout of a matrix.

Possible values are [`wmma_row_major`](@ref), [`wmma_col_major`](@ref) and [`wmma_unspecified`](@ref).
"""
abstract type wmma_fragment_layout end

"""
    wmma_row_major

Type that represents a matrix stored in row major (C style) order.
"""
struct wmma_row_major <: wmma_fragment_layout end

"""
    wmma_col_major

Type that represents a matrix stored in column major (Julia style) order.
"""
struct wmma_col_major <: wmma_fragment_layout end

"""
    wmma_unspecified

Type that represents a matrix stored in an unspecified order.

!!! warning

    This storage format is not valid for all WMMA operations!
"""
struct wmma_unspecified <: wmma_fragment_layout end


export wmma_matrix_a, wmma_matrix_b, wmma_accumulator

abstract type wmma_fragment_use end
struct wmma_matrix_a <: wmma_fragment_use end
struct wmma_matrix_b <: wmma_fragment_use end
struct wmma_accumulator <: wmma_fragment_use end


export wmma_fragment

"""
    wmma_fragment

Type that represents per-thread intermediate results of WMMA operations.

You can access individual elements using the `x` member, but beware that the exact ordering of elements is unspecified.
"""
struct wmma_fragment{M, N, K, FS, T, L <: wmma_fragment_layout, U <: wmma_fragment_use}
    x::NTuple{FS, T}
end

# ------------------
# WMMA configuration
# ------------------

export wmma_config

"""
    wmma_config{M, N, K, d_type}

Type that contains all information for WMMA operations that cannot be inferred from the argument's types.

WMMA instructions calculate the matrix multiply-accumulate operation ``D = A \\cdot B + C``, where ``A`` is a ``M \\times K`` matrix,
``B`` a ``K \\times N`` matrix, and ``C`` and ``D`` are ``M \\times N`` matrices.

`d_type` refers to the type of the elements of matrix ``D``, and can be either `Float16` or `Float32`.

All WMMA operations take a `wmma_config` as their final argument.

# Examples
```jldoctest
julia> config = wmma_config{16, 16, 16, Float32}
wmma_config{16,16,16,Float32}
```
"""
struct wmma_config{M, N, K, d_type} end

# ---------
# Constants
# ---------

map_matrix_to_use = Dict(
                      "a" => wmma_matrix_a,
                      "b" => wmma_matrix_b,
                      "c" => wmma_accumulator,
                      "d" => wmma_accumulator
                        )

map_address_space_to_ty = Dict(
                               "" => AS.Generic,
                               "shared" => AS.Shared,
                               "global" => AS.Global
                              )

# Maps matrix & PTX types to number of elements (size after flattening)
map_num_elements = Dict(
                      "a.f16" => 16,
                      "b.f16" => 16,
                      "c.f16" => 8,
                      "c.f32" => 8,
                      "d.f16" => 8,
                      "d.f32" => 8
                       )

# ----------------
# Helper functions
# ----------------

get_matrix_use(mat) = map_matrix_to_use[mat]
get_address_space(as) = map_address_space_to_ty[as]

get_hl_frag_info(matrix, ptx_el_type) = (
        map_ptx_to_jl_array[ptx_el_type],
        map_ptx_to_jl_frag[ptx_el_type],
        map_frag_sizes["$matrix.$ptx_el_type"],
        map_num_elements["$matrix.$ptx_el_type"]
        )

# ---------
# WMMA load
# ---------

export wmma_load_a, wmma_load_b, wmma_load_c

"""
    wmma_load_a(addr, stride, layout, config)
    wmma_load_b(addr, stride, layout, config)
    wmma_load_c(addr, stride, layout, config)

Load the matrix `a`, `b` or `c` from the memory location indicated by `addr`, and return the resulting [`wmma_fragment`](@ref).

# Arguments
- `addr`: The address to load the matrix from.
- `stride`: The leading dimension of the matrix pointed to by `addr`, specified in number of elements.
- `layout`: The storage layout of the matrix. Possible values are [`wmma_row_major`](@ref) and [`wmma_col_major`](@ref).
- `config`: The WMMA configuration that should be used for loading this matrix. See [`wmma_config`](@ref).

See also: [`wmma_fragment`](@ref), [`wmma_fragment_layout`](@ref), [`wmma_config`](@ref)

!!! warning

    All threads in a warp **MUST** execute the load operation in lockstep, and have to use exactly the same arguments.
    Failure to do so will result in undefined behaviour.
"""
wmma_load_a, wmma_load_b, wmma_load_c

for mat in ["a", "b", "c"],
    layout in ["col", "row"],
    shape in ["m16n16k16"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"],
    elem_type in ["f16", "f32"]


    # Float32 is only supported for C
    if (elem_type == "f32") && (mat != "c")
        continue
    end

    # Name of Julia function
    func_name = Symbol("wmma_load_$mat")

    # Name of the Julia wrapper
    wrapper = Symbol(join_nonempty("llvm", "wmma", "load", mat, layout, shape, addr_space, stride, elem_type, "_"))

    # Get fragment size
    arr_ty, _, _, sz = get_hl_frag_info(mat, elem_type)

    # Get matrix use type
    matrix_use = get_matrix_use(mat)

    # Get layout type
    layout_ty = (layout == "col") ? wmma_col_major : wmma_row_major
    layout_frag_ty = (mat == "c") ? wmma_unspecified : layout_ty

    # Get address space type
    as_ty = get_address_space(addr_space)

    @eval function $func_name(addr::DevicePtr{$arr_ty, $as_ty},
                              stride::Number,
                              layout::Type{$layout_ty},
                              config::Type{wmma_config{16, 16, 16, d_type}}) where d_type
        x = flatten($wrapper(addr, stride))
        return wmma_fragment{16, 16, 16, $sz, $arr_ty, $layout_frag_ty, $matrix_use}(x)
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

- `a`: The [`wmma_fragment`](@ref) corresponding to the matrix ``A``.
- `b`: The [`wmma_fragment`](@ref) corresponding to the matrix ``B``.
- `c`: The [`wmma_fragment`](@ref) corresponding to the matrix ``C``.
- `conf`: The [`wmma_config`](@ref) that should be used in this WMMA operation.

!!! warning

    All threads in a warp **MUST** execute the `mma` operation in lockstep, and have to use exactly the same arguments.
    Failure to do so will result in undefined behaviour.
"""
wmma_mma

for a_layout in ["col", "row"],
    b_layout in ["col", "row"],
    shape in ["m16n16k16"],
    d_elem_type in ["f16", "f32"],
    c_elem_type in ["f16", "f32"],
    b_elem_type in ["f16"],
    a_elem_type in ["f16"]

    # Name of the Julia wrapper
    wrapper = Symbol(join_nonempty("llvm", "wmma", "mma", a_layout, b_layout, shape, d_elem_type, c_elem_type, "_"))

    # Get types
    a_arr_ty, a_frag_ty, a_sz_unfl, a_sz = get_hl_frag_info("a", a_elem_type)
    a_layout_ty = (a_layout == "col") ? wmma_col_major : wmma_row_major

    b_arr_ty, b_frag_ty, b_sz_unfl, b_sz = get_hl_frag_info("b", b_elem_type)
    b_layout_ty = (b_layout == "col") ? wmma_col_major : wmma_row_major

    c_arr_ty, c_frag_ty, c_sz_unfl, c_sz = get_hl_frag_info("c", c_elem_type)

    d_arr_ty, _, _, d_sz = get_hl_frag_info("d", d_elem_type)

    @eval function wmma_mma(a::wmma_fragment{16, 16, 16, $a_sz, $a_arr_ty, $a_layout_ty, wmma_matrix_a},
                            b::wmma_fragment{16, 16, 16, $b_sz, $b_arr_ty, $b_layout_ty, wmma_matrix_b},
                            c::wmma_fragment{16, 16, 16, $c_sz, $c_arr_ty, wmma_unspecified, wmma_accumulator},
                            conf::Type{wmma_config{16, 16, 16, $d_arr_ty}})
        a_unfl = unflatten(NTuple{$a_sz_unfl, $a_frag_ty}, a.x)
        b_unfl = unflatten(NTuple{$b_sz_unfl, $b_frag_ty}, b.x)
        c_unfl = unflatten(NTuple{$c_sz_unfl, $c_frag_ty}, c.x)
        x = flatten($wrapper(a_unfl, b_unfl, c_unfl))
        return wmma_fragment{16, 16, 16, $d_sz, $d_arr_ty, wmma_unspecified, wmma_accumulator}(x)
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
- `d`: The [`wmma_fragment`](@ref) corresponding to the `d` matrix.
- `stride`: The leading dimension of the matrix pointed to by `addr`, specified in number of elements.
- `layout`: The storage layout of the matrix. Possible values are [`wmma_row_major`](@ref) and [`wmma_col_major`](@ref).
- `config`: The WMMA configuration that should be used for storing this matrix. See [`wmma_config`](@ref).

See also: [`wmma_fragment`](@ref), [`wmma_fragment_layout`](@ref), [`wmma_config`](@ref)

!!! warning

    All threads in a warp **MUST** execute the `store` operation in lockstep, and have to use exactly the same arguments.
    Failure to do so will result in undefined behaviour.
"""
wmma_store_d

for mat in ["d"],
    layout in ["col", "row"],
    shape in ["m16n16k16"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"],
    elem_type in ["f16", "f32"]

    # Name of Julia function
    func_name = Symbol("wmma_store_$mat")

    # Name of the Julia wrapper
    wrapper = Symbol(join_nonempty("llvm", "wmma", "store", mat, layout, shape, addr_space, stride, elem_type, "_"))

    # Get types
    arr_ty, frag_ty, sz_unfl, sz = get_hl_frag_info(mat, elem_type)

    # Get matrix use type
    matrix_use = get_matrix_use(mat)

    # Get layout type
    layout_ty = (layout == "col") ? wmma_col_major : wmma_row_major
    layout_frag_ty = wmma_unspecified

    # Get address space type
    as_ty = get_address_space(addr_space)

    @eval function $func_name(addr::DevicePtr{$arr_ty, $as_ty},
                              d::wmma_fragment{16, 16, 16, $sz, $arr_ty, $layout_frag_ty, $matrix_use},
                              stride::Number,
                              layout::Type{$layout_ty},
                              config::Type{wmma_config{16, 16, 16, d_type}}) where d_type
        d_unfl = unflatten(NTuple{$sz_unfl, $frag_ty}, d.x)
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

Return a [`wmma_fragment`](@ref) filled with the value `value`.

This operation is useful if you want to implement a matrix multiplication (and thus want to set ``C = O``).

# Arguments
- `value`: The value used to fill the fragment. Can be a `Float16` or `Float32`.
- `config`: The WMMA configuration that should be used for this WMMA operation. See [`wmma_config`](@ref).
"""
wmma_fill_c

for mat in ["c"],
    elem_type in ["f16", "f32"]

    # Name of the Julia function
    func_name = Symbol("wmma_fill_$mat")

    # Get fragment types and size
    arr_ty, _, _, sz = get_hl_frag_info(mat, elem_type)

    @eval function $func_name(value::$arr_ty,
                              config::Type{wmma_config{M, N, K, d_type}}) where {M, N, K, d_type}

        x = ntuple(i -> value, $sz)
        return wmma_fragment{16, 16, 16, $sz, $arr_ty, wmma_unspecified, wmma_accumulator}(x)
    end
end