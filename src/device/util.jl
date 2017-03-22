export @narrow


"""
Narrow integer and floating-point literals to their smallest representable type.

This is a hack, awaiting a proper solution for #25.
"""
macro narrow(ex)
    esc(visit(ex))
end

visit(ex) = ex
function visit(ex::Expr)
    # some special AST nodes we probably shouldn't convert
    if ex.head == :line
        return ex
    end
    
    return Expr(ex.head, map(visit, ex.args)...)
end
function visit(ex::Integer)
    # NOTE: no automatic subtype enumeration, or we'd convert to Bools
    for T in [UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, UInt128, Int128]
        if typemin(T) <= ex <= typemax(T)
            return convert(T, ex)
        end
    end

    # don't know how to handle this
    return ex
end
function visit(ex::AbstractFloat)
    for T in [Float16, Float32, Float64]
        narrow = convert(T, ex)
        if ex == narrow
            return narrow
        end
    end
    
    # don't know how to handle this
    return ex
end
