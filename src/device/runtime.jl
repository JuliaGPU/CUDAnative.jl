# CUDAnative runtime library
#
# This module defines method instances that will be compiled into a device-specific image
# and will be available to the CUDAnative compiler to call after Julia has generated code.
#
# Most functions implement, or are used to support Julia runtime functions that are expected
# by the Julia compiler to be available at run time, e.g., to dynamically allocate memory,
# box values, etc.

module Runtime

using ..CUDAnative
using LLVM
using LLVM.Interop

import ..CUDAnative: GCFrame
## representation of a runtime method instance

struct RuntimeMethodInstance
    def::Function

    return_type::Type
    types::Tuple
    name::Symbol

    # LLVM types cannot be cached, so we can't put them in the runtime method instance.
    # the actual types are constructed upon accessing them, based on a sentinel value:
    #  - nothing: construct the LLVM type based on its Julia counterparts
    #  - function: call this generator to get the type (when more control is needed)
    llvm_return_type::Union{Nothing, Function}
    llvm_types::Union{Nothing, Function}
    llvm_name::String
end

function Base.getproperty(rt::RuntimeMethodInstance, field::Symbol)
    value = getfield(rt, field)
    if field == :llvm_types
        if value == nothing
            LLVMType[convert.(LLVMType, typ) for typ in rt.types]
        else
            value()
        end
    elseif field == :llvm_return_type
        if value == nothing
            convert.(LLVMType, rt.return_type)
        else
            value()
        end
    else
        return value
    end
end

const methods = Dict{Symbol,RuntimeMethodInstance}()
get(name::Symbol) = methods[name]

# Register a Julia function `def` as a runtime library function identified by `name`. The
# function will be compiled upon first use for argument types `types` and should return
# `return_type`. Use `Runtime.get(name)` to get a reference to this method instance.
#
# The corresponding LLVM types `llvm_types` and `llvm_return_type` will be deduced from
# their Julia counterparts. To influence that conversion, pass a callable object instead;
# this object will be evaluated at run-time and the returned value will be used instead.
#
# When generating multiple runtime functions from a single definition, make sure to specify
# different values for `name`. The LLVM function name will be deduced from that name, but
# you can always specify `llvm_name` to influence that. Never use an LLVM name that starts
# with `julia_` or the function might clash with other compiled functions.
function compile(def, return_type, types, llvm_return_type=nothing, llvm_types=nothing;
                 name=typeof(def).name.mt.name, llvm_name="ptx_$name")
    meth = RuntimeMethodInstance(def,
                                 return_type, types, name,
                                 llvm_return_type, llvm_types, llvm_name)
    if haskey(methods, name)
        error("Runtime function $name has already been registered!")
    end
    methods[name] = meth
    meth
end


## exception handling

function report_exception(ex)
    @cuprintf("""
        ERROR: a %s was thrown during kernel execution.
               Run Julia on debug level 2 for device stack traces.
        """, ex)
    return
end

compile(report_exception, Nothing, (Ptr{Cchar},))

function report_exception_name(ex)
    @cuprintf("""
        ERROR: a %s was thrown during kernel execution.
        Stacktrace:
        """, ex)
    return
end

function report_exception_frame(idx, func, file, line)
    @cuprintf(" [%i] %s at %s:%i\n", idx, func, file, line)
    return
end

compile(report_exception_frame, Nothing, (Cint, Ptr{Cchar}, Ptr{Cchar}, Cint))
compile(report_exception_name, Nothing, (Ptr{Cchar},))

# NOTE: no throw functions are provided here, but replaced by an LLVM pass instead
#       in order to provide some debug information without stack unwinding.


## GC

@enum AddressSpace begin
    Generic         = 1
    Tracked         = 10
    Derived         = 11
    CalleeRooted    = 12
    Loaded          = 13
end

# LLVM type of a tracked pointer
function T_prjlvalue()
    T_pjlvalue = convert(LLVMType, Any, true)
    LLVM.PointerType(eltype(T_pjlvalue), Tracked)
end

# A function that gets replaced by the proper 'malloc' implementation
# for the context it executes in. When the GC is used, calls to this
# function are replaced with 'gc_malloc'; otherwise, this function gets
# rewritten as a call to the allocator, probably 'malloc'.
@generated function managed_malloc(sz::Csize_t)
    T_pint8 = LLVM.PointerType(LLVM.Int8Type(JuliaContext()))
    T_size = convert(LLVMType, Csize_t)
    T_ptr = convert(LLVMType, Ptr{UInt8})

    # create function
    llvm_f, _ = create_function(T_ptr, [T_size])
    mod = LLVM.parent(llvm_f)

    intr = LLVM.Function(mod, "julia.managed_malloc", LLVM.FunctionType(T_pint8, [T_size]))

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)
        ptr = call!(builder, intr, [parameters(llvm_f)[1]])
        jlptr = ptrtoint!(builder, ptr, T_ptr)
        ret!(builder, jlptr)
    end

    call_function(llvm_f, Ptr{UInt8}, Tuple{Csize_t}, :((sz,)))
end

function gc_pool_alloc(sz::Csize_t)
    ptr = malloc(sz)
    if ptr == C_NULL
        @cuprintf("ERROR: Out of dynamic GPU memory (trying to allocate %i bytes)\n", sz)
        throw(OutOfMemoryError())
    end
    return unsafe_pointer_to_objref(ptr)
end

compile(gc_pool_alloc, Any, (Csize_t,), T_prjlvalue)

## boxing and unboxing

const tag_type = UInt
const tag_size = sizeof(tag_type)

const gc_bits = 0x3 # FIXME

# get the type tag of a type at run-time
@generated function type_tag(::Val{type_name}) where type_name
    T_tag = convert(LLVMType, tag_type)
    T_ptag = LLVM.PointerType(T_tag)

    T_pjlvalue = convert(LLVMType, Any, true)

    # create function
    llvm_f, _ = create_function(T_tag)
    mod = LLVM.parent(llvm_f)

    # this isn't really a function, but we abuse it to get the JIT to resolve the address
    typ = LLVM.Function(mod, "jl_" * String(type_name) * "_type",
                        LLVM.FunctionType(T_pjlvalue))

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        typ_var = bitcast!(builder, typ, T_ptag)

        tag = load!(builder, typ_var)

        ret!(builder, tag)
    end

    call_function(llvm_f, tag_type)
end

# we use `jl_value_ptr`, a Julia pseudo-intrinsic that can be used to box and unbox values

@generated function box(val, ::Val{type_name}) where type_name
    sz = sizeof(val)
    allocsz = sz + tag_size

    # type-tags are ephemeral, so look them up at run time
    #tag = unsafe_load(convert(Ptr{tag_type}, type_name))
    tag = :( type_tag(Val(type_name)) )

    quote
        Base.@_inline_meta

        ptr = malloc($(Csize_t(allocsz)))

        # store the type tag
        ptr = convert(Ptr{tag_type}, ptr)
        Core.Intrinsics.pointerset(ptr, $tag | $gc_bits, #=index=# 1, #=align=# $tag_size)

        # store the value
        ptr = convert(Ptr{$val}, ptr+tag_size)
        Core.Intrinsics.pointerset(ptr, val, #=index=# 1, #=align=# $sz)

        unsafe_pointer_to_objref(ptr)
    end
end

@inline function unbox(obj, ::Type{T}) where T
    ptr = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), obj)

    # load the value
    ptr = convert(Ptr{T}, ptr)
    Core.Intrinsics.pointerref(ptr, #=index=# 1, #=align=# sizeof(T))
end

# generate functions functions that exist in the Julia runtime (see julia/src/datatype.c)
for (T, t) in [Int8   => :int8,  Int16  => :int16,  Int32  => :int32,  Int64  => :int64,
               UInt8  => :uint8, UInt16 => :uint16, UInt32 => :uint32, UInt64 => :uint64]
    box_fn   = Symbol("box_$t")
    unbox_fn = Symbol("unbox_$t")
    @eval begin
        $box_fn(val)   = box($T(val), Val($(QuoteNode(t))))
        $unbox_fn(obj) = unbox(obj, $T)

        compile($box_fn, Any, ($T,), T_prjlvalue; llvm_name=$"jl_$box_fn")
        compile($unbox_fn, $T, (Any,); llvm_name=$"jl_$unbox_fn")
    end
end

## Garbage collection

# LLVM type of a pointer to a tracked pointer
function T_pprjlvalue()
    T_pjlvalue = convert(LLVMType, Any, true)
    LLVM.PointerType(
        LLVM.PointerType(eltype(T_pjlvalue), Tracked))
end

# Include GC memory allocation functions into the runtime.
compile(CUDAnative.gc_malloc, Ptr{UInt8}, (Csize_t,))
compile(CUDAnative.gc_malloc_object, Any, (Csize_t,), T_prjlvalue)

# Include GC frame management functions into the runtime.
compile(CUDAnative.new_gc_frame, Any, (Cuint,), T_pprjlvalue)

compile(
    CUDAnative.push_gc_frame,
    Nothing,
    (GCFrame, Cuint),
    () -> convert(LLVMType, Cvoid),
    () -> [T_pprjlvalue(), convert(LLVMType, UInt32)])

compile(
    CUDAnative.pop_gc_frame,
    Nothing,
    (GCFrame,),
    () -> convert(LLVMType, Cvoid),
    () -> [T_pprjlvalue()])

# Also import the safepoint and perma-safepoint functions.
compile(CUDAnative.gc_safepoint, Cvoid, ())
compile(CUDAnative.gc_perma_safepoint, Cvoid, ())

## Arrays

# A data structure that carefully mirrors an in-memory array control
# structure for Julia arrays, as laid out by the compiler.
mutable struct Array1D
    # This is the data layout for Julia arrays, which we adhere to here.
    # 
    #     JL_EXTENSION typedef struct {
    #       JL_DATA_TYPE
    #       void *data;
    #     #ifdef STORE_ARRAY_LEN
    #       size_t length;
    #     #endif
    #       jl_array_flags_t flags;
    #       uint16_t elsize;
    #       uint32_t offset;  // for 1-d only. does not need to get big.
    #       size_t nrows;
    #       union {
    #           // 1d
    #           size_t maxsize;
    #           // Nd
    #           size_t ncols;
    #       };
    #       // other dim sizes go here for ndims > 2
    #
    #       // followed by alignment padding and inline data, or owner pointer
    #     } jl_array_t;

    data::Ptr{UInt8}
    length::Csize_t
    flags::UInt16
    elsize::UInt16
    offset::UInt32
    nrows::Csize_t
    maxsize::Csize_t
end

function zero_fill!(ptr::Ptr{UInt8}, count::Integer)
    for i in 1:count
        unsafe_store!(ptr, UInt8(0), count)
    end
    return
end

function memmove!(dst::Ptr{UInt8}, src::Ptr{UInt8}, sz::Integer)
    if src < dst
        for i in 1:sz
            unsafe_store!(dst, unsafe_load(src, i), i)
        end
    else
        for i in sz:-1:1
            unsafe_store!(dst, unsafe_load(src, i), i)
        end
    end
end

# Resize the buffer to a max size of `newlen`
# The buffer can either be newly allocated or realloc'd, the return
# value is true if a new buffer is allocated and false if it is realloc'd.
# the caller needs to take care of moving the data from the old buffer
# to the new one if necessary.
# When this function returns, the `.data` pointer always points to
# the **beginning** of the new buffer.
function array_resize_buffer(a::Array1D, newlen::Csize_t)::Bool
    elsz = Csize_t(a.elsize)
    nbytes = newlen * elsz
    oldnbytes = a.maxsize * elsz

    if elsz == 1
        nbytes += 1
        oldnbytes += 1
    end

    # Allocate a new buffer. 'managed_malloc' will get replaced with
    # the "right" allocation function for the environment in which this
    # function is compiled. So if the GC is enabled, then 'managed_malloc'
    # will actually call 'gc_malloc'; otherwise, it's probably going to
    # be 'malloc'.
    a.data = managed_malloc(nbytes)
    zero_fill!(a.data + oldnbytes, nbytes - oldnbytes)
    a.maxsize = newlen
    return true
end

function jl_array_grow_at_end(a::Array1D, idx::Csize_t, inc::Csize_t, n::Csize_t)
    data = a.data
    elsz = Csize_t(a.elsize)
    reqmaxsize = a.offset + n + inc
    has_gap = n > idx
    if reqmaxsize > a.maxsize
        nb1 = idx * elsz
        nbinc = inc * elsz

        if reqmaxsize < 4
            newmaxsize = Csize_t(4)
        elseif reqmaxsize >= a.maxsize * 2
            newmaxsize = reqmaxsize
        else
            newmaxsize = a.maxsize * 2
        end

        newbuf = array_resize_buffer(a, newmaxsize)
        newdata = a.data + a.offset * elsz
        if newbuf
            memmove!(newdata, data, nb1)
            if has_gap
                memmove!(newdata + nb1 + nbinc, data + nb1, n * elsz - nb1)
            end
        elseif has_gap
            memmove!(newdata + nb1 + nbinc, newdata + nb1, n * elsz - nb1)
        end
        a.data = data = newdata
    end

    newnrows = n + inc
    a.length = newnrows
    a.nrows = newnrows
    zero_fill!(data + idx * elsz, inc * elsz)
    return
end

function jl_array_grow_end(a::Array1D, inc::Csize_t)
    n = a.nrows
    jl_array_grow_at_end(a, n, inc, n)
    return
end

compile(
    jl_array_grow_end,
    Cvoid,
    (Array1D, Csize_t),
    () -> convert(LLVMType, Cvoid),
    () -> [T_prjlvalue(), convert(LLVMType, Csize_t)])

end
