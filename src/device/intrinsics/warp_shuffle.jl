# Warp Shuffle (B.14)

# TODO: does not work on sub-word (ie. Int16) or non-word divisible sized types

# TODO: should shfl_idx conform to 1-based indexing?

# TODO: these functions should dispatch based on the actual warp size
const ws = Int32(32)


# single-word

# NOTE: CUDA C disagrees with PTX on how shuffles are called
for (name, mode, mask) in ((:shfl_up,   :up,   UInt32(0x00)),
                           (:shfl_down, :down, UInt32(0x1f)),
                           (:shfl_xor,  :bfly, UInt32(0x1f)),
                           (:shfl,      :idx,  UInt32(0x1f)))
    pack_expr = :((($ws - convert(UInt32, width)) << 8) | $mask)
    intrinsic = Symbol("llvm.nvvm.shfl.$mode.i32")

    @eval begin
        export $name
        @inline $name(val::UInt32, srclane::Integer, width::Integer=$ws) =
            ccall($"$intrinsic", llvmcall, UInt32,
                  (UInt32, UInt32, UInt32), val, convert(UInt32, srclane), $pack_expr)
    end
end

## synonyms
# TODO: these reinterpret and reuse the i32 shuffle, should we use the f32 shuffle?
for typ in (Int32, Float32),
    name in [:shfl_up, :shfl_down, :shfl_xor, :shfl]
    @eval @inline $name(val::$typ, srclane, width::Integer=$ws) =
        reinterpret($typ, $name(reinterpret(UInt32, val), srclane, width))
end


# multi-word

## extract a word from a value
@generated function extract_word(val, ::Val{i}) where {i}
    T_int32 = LLVM.Int32Type(jlctx[])

    bytes = Core.sizeof(val)
    T_val = convert(LLVMType, val)
    T_int = LLVM.IntType(8*bytes, jlctx[])

    # create function
    llvmf = create_llvmf(T_int32, [T_val])
    mod = LLVM.parent(llvmf)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvmf, "entry", jlctx[])
        position!(builder, entry)

        equiv = bitcast!(builder, parameters(llvmf)[1], T_int)
        shifted = lshr!(builder, equiv, LLVM.ConstantInt(T_int, 32*(i-1)))
        # extracted = and!(builder, shifted, 2^32-1)
        extracted = trunc!(builder, shifted, T_int32, "word$i")

        ret!(builder, extracted)
    end

    call_llvmf(llvmf, UInt32, Tuple{val}, :( (val,) ))
end

## insert a word into a value
@generated function insert_word(val, word::UInt32, ::Val{i}) where {i}
    T_int32 = LLVM.Int32Type(jlctx[])

    bytes = Core.sizeof(val)
    T_val = convert(LLVMType, val)
    T_int = LLVM.IntType(8*bytes, jlctx[])

    # create function
    llvmf = create_llvmf(T_val, [T_val, T_int32])
    mod = LLVM.parent(llvmf)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvmf, "entry", jlctx[])
        position!(builder, entry)

        equiv = bitcast!(builder, parameters(llvmf)[1], T_int)
        ext = zext!(builder, parameters(llvmf)[2], T_int)
        shifted = shl!(builder, ext, LLVM.ConstantInt(T_int, 32*(i-1)))
        inserted = or!(builder, equiv, shifted)
        orig = bitcast!(builder, inserted, T_val)

        ret!(builder, orig)
    end

    call_llvmf(llvmf, val, Tuple{val, UInt32}, :( (val, word) ))
end

## generic shuffle working on individual words
@generated function _shuffle(op::Function, val::Real, srclane::Integer, width::Integer=ws)
    ex = quote
        Base.@_inline_meta
    end

    # disassemble into words
    words = Symbol[]
    for i in 1:Core.sizeof(val)÷4
        word = Symbol("word$i")
        push!(ex.args, :( $word = extract_word(val, Val{$i}()) ))
        push!(words, word)
    end

    # shuffle
    for word in words
        push!(ex.args, :( $word = op($word, srclane, width)) )
    end

    # reassemble
    push!(ex.args, :( out = zero(val) ))
    for (i,word) in enumerate(words)
        push!(ex.args, :( out = insert_word(out, $word, Val{$i}()) ))
    end

    push!(ex.args, :( out ))
    return ex
end

## define entry-point functions
for name in [:shfl_up, :shfl_down, :shfl_xor, :shfl]
    @eval $name(val::Real, srclane, width::Integer=$ws) =
        _shuffle($name, val, srclane, width)
end


# documentation

@doc """
    shfl_idx(val::Real, src::Integer, width::Integer=32)

Shuffle a value from a directly indexed lane `src`
""" shfl

@doc """
    shfl_up(val::Real, src::Integer, width::Integer=32)

Shuffle a value from a lane with lower ID relative to caller.
""" shfl_up

@doc """
    shfl_down(val::Real, src::Integer, width::Integer=32)

Shuffle a value from a lane with higher ID relative to caller.
""" shfl_down

@doc """
    shfl_xor(val::Real, src::Integer, width::Integer=32)

Shuffle a value from a lane based on bitwise XOR of own lane ID.
""" shfl_xor
