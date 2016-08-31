# Native intrinsics

export
    # Indexing and dimensions
    threadIdx, blockDim, blockIdx, gridDim,
    warpsize, nearest_warpsize,

    # Memory management
    sync_threads,
    @cuStaticSharedMem, @cuDynamicSharedMem



#
# Support functionality
#

# TODO: compute capability checks

const llvmtypes = Dict{Type,Symbol}(
    Void    => :void,
    Int32   => :i32,
    Int64   => :i64,
    Float32 => :float,
    Float64 => :double
)

const jltypes = Dict{Symbol,Type}(v => k for (k,v) in llvmtypes)

"""
Decode an expression of the form:

   function(arg::arg_type, arg::arg_type, ... arg::arg_type)::return_type

Returns a tuple containing the function name, a vector of argument, a vector of argument
types and the return type (all in symbolic form).
"""
function decode_call(e)
    @assert e.head == :(::)
    rettype = e.args[2]::Symbol

    call = e.args[1]
    @assert call.head == :call

    fn = Symbol(call.args[1])
    args = Symbol[arg.args[1] for arg in call.args[2:end]]
    argtypes = Symbol[arg.args[2] for arg in call.args[2:end]]

    return fn, args, argtypes, rettype
end

"""
Generate a `llvmcall` statement calling an intrinsic specified as follows:

    intrinsic(arg::arg_type, arg::arg_type, ... arg::arg_type)::return_type [attr]

The argument types should be valid LLVM type identifiers (eg. i32, float, double).
Conversions to the corresponding Julia type are automatically generated; make sure the
actual arguments are of the same type to make these conversions no-ops. The optional
argument `attr` indicates which LLVM function attributes (such as `readnone` or `nounwind`)
to add to the intrinsic declaration.

For example, the following call:
    @wrap __some_intrinsic(x::float, y::double)::float

will yield the following `llvmcall`:

    Base.llvmcall(("declare float @__somme__intr(float, double)",
                   "%3 = call float @__somme__intr(float %0, double %1)
                    ret float %3"),
                  Float32, Tuple{Float32,Float64},
                  convert(Float32,x), convert(Float64,y))
"""
macro wrap(call, attrs="")
    intrinsic, args, argtypes, rettype = decode_call(call)

    llvm_args = String["%$i" for i in 0:length(argtypes)]
    if rettype == :void
        llvm_ret_asgn = ""
        llvm_ret = "void"
    else
        llvm_ret_var = "%$(length(argtypes)+1)"
        llvm_ret_asgn = "$llvm_ret_var = "
        llvm_ret = "$rettype $llvm_ret_var"
    end
    llvm_declargs = join(argtypes, ", ")
    llvm_defargs = join(("$t $arg" for (t,arg) in zip(argtypes, llvm_args)), ", ")

    julia_argtypes = (jltypes[t] for t in argtypes)
    julia_args = (:(convert($argtype, $arg)) for (arg, argtype) in zip(args, julia_argtypes))

    return quote
        Base.llvmcall(
            ($"""declare $rettype @$intrinsic($llvm_declargs)""",
             $"""$llvm_ret_asgn call $rettype @$intrinsic($llvm_defargs)
                 ret $llvm_ret"""),
            $(jltypes[rettype]), Tuple{$(julia_argtypes...)}, $(julia_args...))
    end
end



#
# Indexing and dimensions
#

for dim in (:x, :y, :z)
    # Thread index
    fn = Symbol("threadIdx_$dim")
    @eval @inline @target ptx $fn() = (@wrap llvm.nvvm.read.ptx.sreg.tid.$dim()::i32    "readnone nounwind")+Int32(1)

    # Block size (#threads per block)
    fn = Symbol("blockDim_$dim")
    @eval @inline @target ptx $fn() =  @wrap llvm.nvvm.read.ptx.sreg.ntid.$dim()::i32   "readnone nounwind"

    # Block index
    fn = Symbol("blockIdx_$dim")
    @eval @inline @target ptx $fn() = (@wrap llvm.nvvm.read.ptx.sreg.ctaid.$dim()::i32  "readnone nounwind")+Int32(1)

    # Grid size (#blocks per grid)
    fn = Symbol("gridDim_$dim")
    @eval @inline @target ptx $fn() =  @wrap llvm.nvvm.read.ptx.sreg.nctaid.$dim()::i32 "readnone nounwind"
end

# Tuple accessors
@inline @target ptx threadIdx() = CUDAdrv.CuDim3(threadIdx_x(), threadIdx_y(), threadIdx_z())
@inline @target ptx blockDim() =  CUDAdrv.CuDim3(blockDim_x(),  blockDim_y(),  blockDim_z())
@inline @target ptx blockIdx() =  CUDAdrv.CuDim3(blockIdx_x(),  blockIdx_y(),  blockIdx_z())
@inline @target ptx gridDim() =   CUDAdrv.CuDim3(gridDim_x(),   gridDim_y(),   gridDim_z())

# NOTE: we often need a const warpsize (eg. for shared memory), sp keep this fixed for now
# @inline @target ptx warpsize() = @wrap llvm.nvvm.read.ptx.sreg.warpsize()::i32 "readnone nounwind"
const warpsize = Int32(32)

"Return the nearest multiple of a warpsize, a common requirement for the amount of threads."
@inline nearest_warpsize(threads) =  threads + (warpsize - threads % warpsize) % warpsize



#
# Thread management
#

# Synchronization
# TODO: rename to syncthreads
@inline @target ptx sync_threads() = @wrap llvm.nvvm.barrier0()::void "readnone nounwind"



#
# Shared memory
#

# FIXME: this adds module-scope declarations by means of `llvmcall`, which is unsupported
# TODO: return an Array-like object (containing the number of elements) instead of a raw pointer
# TODO: downcasting pointers to global AS might be inefficient
#       -> check if AS propagation resolves this
#       -> Ptr{AS}, ASPtr{AS}, ...?
# BUG: calling a device function referencing a static-memory @cuSharedMem will reference
#      the same memory -- how does this work in CUDA?

shmem_id = 0

# TODO: shape instead of len

"""
    @cuStaticSharedMem(typ::Type, nel::Integer) -> CuDeviceArray{typ}

Get an array pointing to a statically-allocated piece of shared memory. The type `typ` and
number of elements `nel` should be statically known after type inference, or a (currently
unreported) error will be thrown resulting in a runtime generic call to an internal
generator function.
"""
# TODO: test multi shmem case
macro cuStaticSharedMem(typ, size)
    global shmem_id
    id = shmem_id::Int += 1
    return esc(:(CUDAnative.generate_static_shmem($typ, Val{$size}, Val{$id})))
end

@generated function generate_static_shmem{T,N,I}(::Type{T}, ::Type{Val{N}}, ::Type{Val{I}})
    return emit_static_shmem(T, N, I)
end

emit_static_shmem(jltyp::Type, len::Integer, id::Integer) =
    return emit_static_shmem(jltyp, (len,), id)

function emit_static_shmem{N}(jltyp::Type, shape::NTuple{N,Int}, id::Integer)
    if !haskey(llvmtypes, jltyp)
        error("cuStaticSharedMem: unsupported type '$jltyp'")
    end
    llvmtyp = llvmtypes[jltyp]

    var = Symbol(:@shmem, id)
    len = prod(shape)

    return quote
        CuDeviceArray{$jltyp}($shape, Base.llvmcall(
            ($"""$var = internal addrspace(3) global [$len x $llvmtyp] zeroinitializer, align 4""",
             $"""%1 = getelementptr inbounds [$len x $llvmtyp], [$len x $llvmtyp] addrspace(3)* $var, i64 0, i64 0
                 %2 = addrspacecast $llvmtyp addrspace(3)* %1 to $llvmtyp addrspace(0)*
                 ret $llvmtyp* %2"""),
            Ptr{$jltyp}, Tuple{}))
    end
end


"""
    @cuDynamicSharedMem(typ::Type, nel::Integer, [offset::Integer=0]) -> CuDeviceArray{typ}

Get an array pointing to a dynamically-allocated piece of shared memory. The type `typ`
should be statically known after type inference, or a (currently unreported) error will be
thrown resulting in a runtime generic call to an internal generator function. The necessary
memory needs to be allocated when calling the kernel.

Optionally, an offset parameter indicating how many bytes to add to the base shared memory
pointer can be specified. This is useful when dealing with a heterogeneous buffer of dynamic
shared memory; in the case of a homogeneous multi-part buffer it is preferred to use `view`.
"""
macro cuDynamicSharedMem(typ, size, offset=0)
    global shmem_id
    id = shmem_id::Int += 1
    return esc(:(CUDAnative.generate_dynamic_shmem($typ, $size, $offset, Val{$id})))
end

@generated function generate_dynamic_shmem{T,I}(::Type{T}, size, offset, ::Type{Val{I}})
    return emit_dynamic_shmem(T, :(size), :(offset), I)
end

# TODO: boundscheck against %dynamic_smem_size (currently unsupported by LLVM)
function emit_dynamic_shmem(jltyp::Type, size::Union{Expr,Symbol}, offset::Symbol, id::Integer)
    if !haskey(llvmtypes, jltyp)
        error("cuDynamicSharedMem: unsupported type '$jltyp'")
    end
    llvmtyp = llvmtypes[jltyp]

    var = Symbol(:@shmem, id)

    return quote
        CuDeviceArray{$jltyp}($size, Base.llvmcall(
            ($"""$var = external addrspace(3) global [0 x $llvmtyp]""",
             $"""%1 = getelementptr inbounds [0 x $llvmtyp], [0 x $llvmtyp] addrspace(3)* $var, i64 0, i64 0
                 %2 = addrspacecast $llvmtyp addrspace(3)* %1 to $llvmtyp addrspace(0)*
                 ret $llvmtyp* %2"""),
            Ptr{$jltyp}, Tuple{}) + $offset)
    end
end

# NOTE: this might be a neater approach (with a user-end macro for hiding the `Val{N}`):

# for typ in ((Int64,   :i64),
#             (Float32, :float),
#             (Float64, :double))
#     T, U = typ
#     @eval begin
#         cuSharedMem{T}(::Type{$T}) = Base.llvmcall(
#             ($"""@shmem_$U = external addrspace(3) global [0 x $U]""",
#              $"""%1 = getelementptr inbounds [0 x $U], [0 x $U] addrspace(3)* @shmem_$U, i64 0, i64 0
#                  %2 = addrspacecast $U addrspace(3)* %1 to $U addrspace(0)*
#                  ret $U* %2"""),
#             Ptr{$T}, Tuple{})
#         cuSharedMem{T,N}(::Type{$T}, ::Val{N}) = Base.llvmcall(
#             ($"""@shmem_$U = internal addrspace(3) global [$N x $llvmtyp] zeroinitializer, align 4""",
#              $"""%1 = getelementptr inbounds [$N x $U], [$N x $U] addrspace(3)* @shmem_$U, i64 0, i64 0
#                  %2 = addrspacecast $U addrspace(3)* %1 to $U addrspace(0)*
#                  ret $U* %2"""),
#             Ptr{$T}, Tuple{})
#     end
# end

# However, it requires a change to `llvmcall`, as now calling the static case twice results in
#          a reference to the same memory



#
# Shuffling
#

# TODO: should shfl_idx conform to 1-based indexing?

## narrow

for typ in ((Int32,   :i32, :i32),
            (UInt32,  :i32, :i32),
            (Float32, :f32, :float))
    jl, intr, llvm = typ

    for op in ((:up,   Int32(0x00)),
               (:down, Int32(0x1f)),
               (:bfly, Int32(0x1f)),
               (:idx,  Int32(0x1f)))
        mode, mask = op
        fname = Symbol("shfl_$mode")
        pack_expr = :(((warpsize - Int32(width)) << 8) | $mask)
        @static if VersionNumber(Base.libllvm_version) >= v"3.9-"
            intrinsic = Symbol("llvm.nvvm.shfl.$mode.$intr")
            @eval begin
                export $fname
                @inline $fname(val::$jl, srclane::Integer, width::Integer=warpsize) = Base.llvmcall(
                        ($"""declare $llvm @$intrinsic($llvm, i32, i32)""",
                         $"""%4 = call $llvm @$intrinsic($llvm %0, i32 %1, i32 %2)
                             ret $llvm %4"""),
                        $jl, Tuple{$jl, Int32, Int32}, val, Int32(srclane),
                        $pack_expr)
            end
        else
            instruction = Symbol("shfl.$mode.b32")  # NOTE: only b32 available, no i32/f32
            @eval begin
                export $fname
                @inline $fname(val::$jl, srclane::Integer, width::Integer=warpsize) = Base.llvmcall(
                        $"""%4 = call $llvm asm sideeffect "$instruction \$0, \$1, \$2, \$3;", "=r,r,r,r"($llvm %0, i32 %1, i32 %2)
                            ret $llvm %4""",    # "
                        $jl, Tuple{$jl, Int32, Int32}, val, Int32(srclane),
                        $pack_expr)
            end
        end
    end
end


## wide

@inline decode(val::UInt64) = trunc(UInt32,  val & 0x00000000ffffffff),
                              trunc(UInt32, (val & 0xffffffff00000000)>>32)

@inline encode(x::UInt32, y::UInt32) = UInt64(x) | UInt64(y)<<32

# NOTE: we only reuse the i32 shuffle, does it make any difference using eg. f32 shuffle for f64 values?
for typ in (Int64, UInt64, Float64)
    for mode in (:up, :down, :bfly, :idx)
        fname = Symbol("shfl_$mode")
        @eval begin
            export $fname
            @inline function $fname(val::$typ, srclane::Integer, width::Integer=warpsize)
                x,y = decode(reinterpret(UInt64, val))
                x = $fname(x, srclane, width)
                y = $fname(y, srclane, width)
                reinterpret($typ, encode(x,y))
            end
        end
    end
end


#
# Math
#

## trigonometric

@inline @target ptx cos(x::Float64) = @wrap __nv_cos(x::double)::double
@inline @target ptx cos(x::Float32) = @wrap __nv_cosf(x::float)::float

@inline @target ptx cospi(x::Float64) = @wrap __nv_cospi(x::double)::double
@inline @target ptx cospi(x::Float32) = @wrap __nv_cospif(x::float)::float

@inline @target ptx sin(x::Float64) = @wrap __nv_sin(x::double)::double
@inline @target ptx sin(x::Float32) = @wrap __nv_sinf(x::float)::float

@inline @target ptx sinpi(x::Float64) = @wrap __nv_sinpi(x::double)::double
@inline @target ptx sinpi(x::Float32) = @wrap __nv_sinpif(x::float)::float

@inline @target ptx tan(x::Float64) = @wrap __nv_tan(x::double)::double
@inline @target ptx tan(x::Float32) = @wrap __nv_tanf(x::float)::float


## inverse trigonometric

@inline @target ptx acos(x::Float64) = @wrap __nv_acos(x::double)::double
@inline @target ptx acos(x::Float32) = @wrap __nv_acosf(x::float)::float

@inline @target ptx asin(x::Float64) = @wrap __nv_asin(x::double)::double
@inline @target ptx asin(x::Float32) = @wrap __nv_asinf(x::float)::float

@inline @target ptx atan(x::Float64) = @wrap __nv_atan(x::double)::double
@inline @target ptx atan(x::Float32) = @wrap __nv_atanf(x::float)::float

@inline @target ptx atan2(x::Float64, y::Float64) = @wrap __nv_atan2(x::double, y::double)::double
@inline @target ptx atan2(x::Float32, y::Float32) = @wrap __nv_atan2f(x::float, y::float)::float


## hyperbolic

@inline @target ptx cosh(x::Float64) = @wrap __nv_cosh(x::double)::double
@inline @target ptx cosh(x::Float32) = @wrap __nv_coshf(x::float)::float

@inline @target ptx sinh(x::Float64) = @wrap __nv_sinh(x::double)::double
@inline @target ptx sinh(x::Float32) = @wrap __nv_sinhf(x::float)::float

@inline @target ptx tanh(x::Float64) = @wrap __nv_tanh(x::double)::double
@inline @target ptx tanh(x::Float32) = @wrap __nv_tanhf(x::float)::float


## inverse hyperbolic

@inline @target ptx acosh(x::Float64) = @wrap __nv_acosh(x::double)::double
@inline @target ptx acosh(x::Float32) = @wrap __nv_acoshf(x::float)::float

@inline @target ptx asinh(x::Float64) = @wrap __nv_asinh(x::double)::double
@inline @target ptx asinh(x::Float32) = @wrap __nv_asinhf(x::float)::float

@inline @target ptx atanh(x::Float64) = @wrap __nv_atanh(x::double)::double
@inline @target ptx atanh(x::Float32) = @wrap __nv_atanhf(x::float)::float


## logarithmic

@inline @target ptx log(x::Float64) = @wrap __nv_log(x::double)::double
@inline @target ptx log(x::Float32) = @wrap __nv_logf(x::float)::float

@inline @target ptx log10(x::Float64) = @wrap __nv_log10(x::double)::double
@inline @target ptx log10(x::Float32) = @wrap __nv_log10f(x::float)::float

@inline @target ptx log1p(x::Float64) = @wrap __nv_log1p(x::double)::double
@inline @target ptx log1p(x::Float32) = @wrap __nv_log1pf(x::float)::float

@inline @target ptx log2(x::Float64) = @wrap __nv_log2(x::double)::double
@inline @target ptx log2(x::Float32) = @wrap __nv_log2f(x::float)::float

@inline @target ptx logb(x::Float64) = @wrap __nv_logb(x::double)::double
@inline @target ptx logb(x::Float32) = @wrap __nv_logbf(x::float)::float

@inline @target ptx ilogb(x::Float64) = @wrap __nv_ilogb(x::double)::i32
@inline @target ptx ilogb(x::Float32) = @wrap __nv_ilogbf(x::float)::i32


## exponential

@inline @target ptx exp(x::Float64) = @wrap __nv_exp(x::double)::double
@inline @target ptx exp(x::Float32) = @wrap __nv_expf(x::float)::float

@inline @target ptx exp2(x::Float64) = @wrap __nv_exp2(x::double)::double
@inline @target ptx exp2(x::Float32) = @wrap __nv_exp2f(x::float)::float

@inline @target ptx exp10(x::Float64) = @wrap __nv_exp10(x::double)::double
@inline @target ptx exp10(x::Float32) = @wrap __nv_exp10f(x::float)::float

@inline @target ptx expm1(x::Float64) = @wrap __nv_expm1(x::double)::double
@inline @target ptx expm1(x::Float32) = @wrap __nv_expm1f(x::float)::float

@inline @target ptx ldexp(x::Float64, y::Int32) = @wrap __nv_ldexp(x::double, y::i32)::double
@inline @target ptx ldexp(x::Float32, y::Int32) = @wrap __nv_ldexpf(x::float, y::i32)::float


## error

@inline @target ptx erf(x::Float64) = @wrap __nv_erf(x::double)::double
@inline @target ptx erf(x::Float32) = @wrap __nv_erff(x::float)::float

@inline @target ptx erfinv(x::Float64) = @wrap __nv_erfinv(x::double)::double
@inline @target ptx erfinv(x::Float32) = @wrap __nv_erfinvf(x::float)::float

@inline @target ptx erfc(x::Float64) = @wrap __nv_erfc(x::double)::double
@inline @target ptx erfc(x::Float32) = @wrap __nv_erfcf(x::float)::float

@inline @target ptx erfcinv(x::Float64) = @wrap __nv_erfcinv(x::double)::double
@inline @target ptx erfcinv(x::Float32) = @wrap __nv_erfcinvf(x::float)::float

@inline @target ptx erfcx(x::Float64) = @wrap __nv_erfcx(x::double)::double
@inline @target ptx erfcx(x::Float32) = @wrap __nv_erfcxf(x::float)::float


## integer handling (bit twiddling)

@inline @target ptx brev(x::Int32) =   @wrap __nv_brev(x::i32)::i32
@inline @target ptx brev(x::Int64) =   @wrap __nv_brevll(x::i64)::i64

@inline @target ptx clz(x::Int32) =   @wrap __nv_clz(x::i32)::i32
@inline @target ptx clz(x::Int64) =   @wrap __nv_clzll(x::i64)::i32

@inline @target ptx ffs(x::Int32) = @wrap __nv_ffs(x::i32)::i32
@inline @target ptx ffs(x::Int64) = @wrap __nv_ffsll(x::i64)::i32

@inline @target ptx byte_perm(x::Int32, y::Int32, z::Int32) = @wrap __nv_byte_perm(x::i32, y::i32, z::i32)::i32

@inline @target ptx popc(x::Int32) = @wrap __nv_popc(x::i32)::i32
@inline @target ptx popc(x::Int64) = @wrap __nv_popcll(x::i64)::i32


## floating-point handling

@inline @target ptx isfinite(x::Float32) = (@wrap __nv_finitef(x::float)::i32) != 0
@inline @target ptx isfinite(x::Float64) = (@wrap __nv_isfinited(x::double)::i32) != 0

@inline @target ptx isinf(x::Float32) = (@wrap __nv_isinfd(x::double)::i32) != 0
@inline @target ptx isinf(x::Float64) = (@wrap __nv_isinff(x::float)::i32) != 0

@inline @target ptx isnan(x::Float32) = (@wrap __nv_isnand(x::double)::i32) != 0
@inline @target ptx isnan(x::Float64) = (@wrap __nv_isnanf(x::float)::i32) != 0

@inline @target ptx nearbyint(x::Float64) = @wrap __nv_nearbyint(x::double)::double
@inline @target ptx nearbyint(x::Float32) = @wrap __nv_nearbyintf(x::float)::float

@inline @target ptx nextafter(x::Float64, y::Float64) = @wrap __nv_nextafter(x::double, y::double)::double
@inline @target ptx nextafter(x::Float32, y::Float32) = @wrap __nv_nextafterf(x::float, y::float)::float


## sign handling

@inline @target ptx signbit(x::Float64) = (@wrap __nv_signbitd(x::double)::i32) != 0
@inline @target ptx signbit(x::Float32) = (@wrap __nv_signbitf(x::float)::i32) != 0

@inline @target ptx copysign(x::Float64, y::Float64) = @wrap __nv_copysign(x::double, y::double)::double
@inline @target ptx copysign(x::Float32, y::Float32) = @wrap __nv_copysignf(x::float, y::float)::float

@inline @target ptx abs(x::Int32) =   @wrap __nv_abs(x::i32)::i32
@inline @target ptx abs(f::Float64) = @wrap __nv_fabs(f::double)::double
@inline @target ptx abs(f::Float32) = @wrap __nv_fabsf(f::float)::float
@inline @target ptx abs(x::Int64) =   @wrap __nv_llabs(x::i64)::i64


## roots and powers

@inline @target ptx sqrt(x::Float64) = @wrap __nv_sqrt(x::double)::double
@inline @target ptx sqrt(x::Float32) = @wrap __nv_sqrtf(x::float)::float

@inline @target ptx rsqrt(x::Float64) = @wrap __nv_rsqrt(x::double)::double
@inline @target ptx rsqrt(x::Float32) = @wrap __nv_rsqrtf(x::float)::float

@inline @target ptx cbrt(x::Float64) = @wrap __nv_cbrt(x::double)::double
@inline @target ptx cbrt(x::Float32) = @wrap __nv_cbrtf(x::float)::float

@inline @target ptx rcbrt(x::Float64) = @wrap __nv_rcbrt(x::double)::double
@inline @target ptx rcbrt(x::Float32) = @wrap __nv_rcbrtf(x::float)::float

@inline @target ptx pow(x::Float64, y::Float64) = @wrap __nv_pow(x::double, y::double)::double
@inline @target ptx pow(x::Float32, y::Float32) = @wrap __nv_powf(x::float, y::float)::float
@inline @target ptx pow(x::Float64, y::Int32) =   @wrap __nv_powi(x::double, y::i32)::double
@inline @target ptx pow(x::Float32, y::Int32) =   @wrap __nv_powif(x::float, y::i32)::float


## rounding and selection

# TODO: differentiate in return type, map correctly
# @inline @target ptx round(x::Float64) = @wrap __nv_llround(x::double)::i64
# @inline @target ptx round(x::Float32) = @wrap __nv_llroundf(x::float)::i64
# @inline @target ptx round(x::Float64) = @wrap __nv_round(x::double)::double
# @inline @target ptx round(x::Float32) = @wrap __nv_roundf(x::float)::float

# TODO: differentiate in return type, map correctly
# @inline @target ptx rint(x::Float64) = @wrap __nv_llrint(x::double)::i64
# @inline @target ptx rint(x::Float32) = @wrap __nv_llrintf(x::float)::i64
# @inline @target ptx rint(x::Float64) = @wrap __nv_rint(x::double)::double
# @inline @target ptx rint(x::Float32) = @wrap __nv_rintf(x::float)::float

# TODO: would conflict with trunc usage in this module
# @inline @target ptx trunc(x::Float64) = @wrap __nv_trunc(x::double)::double
# @inline @target ptx trunc(x::Float32) = @wrap __nv_truncf(x::float)::float

@inline @target ptx ceil(x::Float64) = @wrap __nv_ceil(x::double)::double
@inline @target ptx ceil(x::Float32) = @wrap __nv_ceilf(x::float)::float

@inline @target ptx floor(f::Float64) = @wrap __nv_floor(f::double)::double
@inline @target ptx floor(f::Float32) = @wrap __nv_floorf(f::float)::float

@inline @target ptx min(x::Int32, y::Int32) = @wrap __nv_min(x::i32, y::i32)::i32
@inline @target ptx min(x::Int64, y::Int64) = @wrap __nv_llmin(x::i64, y::i64)::i64
@inline @target ptx min(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umin(x::i32, y::i32)::i32)
@inline @target ptx min(x::UInt64, y::UInt64) = convert(UInt64, @wrap __nv_ullmin(x::i64, y::i64)::i64)
@inline @target ptx min(x::Float64, y::Float64) = @wrap __nv_fmin(x::double, y::double)::double
@inline @target ptx min(x::Float32, y::Float32) = @wrap __nv_fminf(x::float, y::float)::float

@inline @target ptx max(x::Int32, y::Int32) = @wrap __nv_max(x::i32, y::i32)::i32
@inline @target ptx max(x::Int64, y::Int64) = @wrap __nv_llmax(x::i64, y::i64)::i64
@inline @target ptx max(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umax(x::i32, y::i32)::i32)
@inline @target ptx max(x::UInt64, y::UInt64) = convert(UInt64, @wrap __nv_ullmax(x::i64, y::i64)::i64)
@inline @target ptx max(x::Float64, y::Float64) = @wrap __nv_fmax(x::double, y::double)::double
@inline @target ptx max(x::Float32, y::Float32) = @wrap __nv_fmaxf(x::float, y::float)::float

@inline @target ptx saturate(x::Float32) = @wrap __nv_saturatef(x::float)::float


## division and remainder

@inline @target ptx mod(x::Float64, y::Float64) = @wrap __nv_fmod(x::double, y::double)::double
@inline @target ptx mod(x::Float32, y::Float32) = @wrap __nv_fmodf(x::float, y::float)::float

@inline @target ptx rem(x::Float64, y::Float64) = @wrap __nv_remainder(x::double, y::double)::double
@inline @target ptx rem(x::Float32, y::Float32) = @wrap __nv_remainderf(x::float, y::float)::float


## gamma function

@inline @target ptx lgamma(x::Float64) = @wrap __nv_lgamma(x::double)::double
@inline @target ptx lgamma(x::Float32) = @wrap __nv_lgammaf(x::float)::float

@inline @target ptx tgamma(x::Float64) = @wrap __nv_tgamma(x::double)::double
@inline @target ptx tgamma(x::Float32) = @wrap __nv_tgammaf(x::float)::float


## Bessel

@inline @target ptx j0(x::Float64) = @wrap __nv_j0(x::double)::double
@inline @target ptx j0(x::Float32) = @wrap __nv_j0f(x::float)::float

@inline @target ptx j1(x::Float64) = @wrap __nv_j1(x::double)::double
@inline @target ptx j1(x::Float32) = @wrap __nv_j1f(x::float)::float

@inline @target ptx jn(n::Int32, x::Float64) = @wrap __nv_jn(n::i32, x::double)::double
@inline @target ptx jn(n::Int32, x::Float32) = @wrap __nv_jnf(n::i32, x::float)::float

@inline @target ptx y0(x::Float64) = @wrap __nv_y0(x::double)::double
@inline @target ptx y0(x::Float32) = @wrap __nv_y0f(x::float)::float

@inline @target ptx y1(x::Float64) = @wrap __nv_y1(x::double)::double
@inline @target ptx y1(x::Float32) = @wrap __nv_y1f(x::float)::float

@inline @target ptx yn(n::Int32, x::Float64) = @wrap __nv_yn(n::i32, x::double)::double
@inline @target ptx yn(n::Int32, x::Float32) = @wrap __nv_ynf(n::i32, x::float)::float



## distributions

@inline @target ptx normcdf(x::Float64) = @wrap __nv_normcdf(x::double)::double
@inline @target ptx normcdf(x::Float32) = @wrap __nv_normcdff(x::float)::float

@inline @target ptx normcdfinv(x::Float64) = @wrap __nv_normcdfinv(x::double)::double
@inline @target ptx normcdfinv(x::Float32) = @wrap __nv_normcdfinvf(x::float)::float



#
# Unsorted
#

@inline @target ptx hypot(x::Float64, y::Float64) = @wrap __nv_hypot(x::double, y::double)::double
@inline @target ptx hypot(x::Float32, y::Float32) = @wrap __nv_hypotf(x::float, y::float)::float

@inline @target ptx fma(x::Float64, y::Float64, z::Float64) = @wrap __nv_fma(x::double, y::double, z::double)::double
@inline @target ptx fma(x::Float32, y::Float32, z::Float32) = @wrap __nv_fmaf(x::float, y::float, z::float)::float

@inline @target ptx sad(x::Int32, y::Int32, z::Int32) = @wrap __nv_sad(x::i32, y::i32, z::i32)::i32
@inline @target ptx sad(x::UInt32, y::UInt32, z::UInt32) = convert(UInt32, @wrap __nv_usad(x::i32, y::i32, z::i32)::i32)

@inline @target ptx dim(x::Float64, y::Float64) = @wrap __nv_fdim(x::double, y::double)::double
@inline @target ptx dim(x::Float32, y::Float32) = @wrap __nv_fdimf(x::float, y::float)::float

@inline @target ptx mul24(x::Int32, y::Int32) = @wrap __nv_mul24(x::i32, y::i32)::i32
@inline @target ptx mul24(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umul24(x::i32, y::i32)::i32)

@inline @target ptx mul64hi(x::Int64, y::Int64) = @wrap __nv_mul64hi(x::i64, y::i64)::i64
@inline @target ptx mul64hi(x::UInt64, y::UInt64) = convert(UInt64, @wrap __nv_umul64hi(x::i64, y::i64)::i64)
@inline @target ptx mulhi(x::Int32, y::Int32) = @wrap __nv_mulhi(x::i32, y::i32)::i32
@inline @target ptx mulhi(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umulhi(x::i32, y::i32)::i32)

@inline @target ptx hadd(x::Int32, y::Int32) = @wrap __nv_hadd(x::i32, y::i32)::i32
@inline @target ptx hadd(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_uhadd(x::i32, y::i32)::i32)

@inline @target ptx rhadd(x::Int32, y::Int32) = @wrap __nv_rhadd(x::i32, y::i32)::i32
@inline @target ptx rhadd(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_urhadd(x::i32, y::i32)::i32)

@inline @target ptx scalbn(x::Float64, y::Int32) = @wrap __nv_scalbn(x::double, y::i32)::double
@inline @target ptx scalbn(x::Float32, y::Int32) = @wrap __nv_scalbnf(x::float, y::i32)::float