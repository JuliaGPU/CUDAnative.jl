export @narrow, @narrow32


"""
Narrow integer and floating-point literals to a fixed bit-width representation.

This is a hack, awaiting a proper solution for #25.
"""
macro narrow(width, ex)
    quote
        const Int = $(Symbol(:Int, width))
        const UInt = $(Symbol(:UInt, width))
        $(esc(narrow_code(width, ex)))
    end
end

macro narrow32(ex)
    esc(quote
        @narrow 32 $ex
    end)
end

narrow_code(width, ex) = ex
function narrow_code(width, ex::Expr)
    # some special AST nodes we probably shouldn't convert
    if ex.head == :line || ex.head == :curly
        return ex
    end
    
    return Expr(ex.head, map(arg->narrow_code(width,arg), ex.args)...)
end

const narrow_integers = Dict{Int,Tuple{Type,Type}}(
    8   => (Int8,   UInt8),
    16  => (Int16,  UInt16),
    32  => (Int32,  UInt32),
    64  => (Int64,  UInt64),
    128 => (Int128, UInt128),
)
function narrow_code(width, val::Integer)
    if !haskey(narrow_integers, width)
        error("Cannot narrow $val to $width-wide integer: no conversion known")
    end
    T = narrow_integers[width][isa(val,Signed)?1:2]
    if typemin(T) <= val <= typemax(T)
        return convert(T, val)
    else
        error("Cannot narrow $val to $width-wide integer $T: cannot represent value")
    end
end

const narrow_floats = Dict{Int,Type}(
    16  => Float16,
    32  => Float32,
    64  => Float64,
)
function narrow_code(width, val::AbstractFloat)
    if !haskey(narrow_integers, width)
        error("Cannot narrow $val to $width-wide float: no conversion known")
    end
    T = narrow_floats[width]
    newval = convert(T, val)
    if val == newval
        return newval
    else
        error("Cannot narrow $val to $width-wide float: cannot represent value")
    end
end
