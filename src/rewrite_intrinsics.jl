function rewrite_expr_cu(m::LazyMethod, expr)
    was_rewritten = false
    body = first(Sugar.replace_expr(expr) do expr
        if isa(expr, Expr)
            args, head = expr.args, expr.head
            if head == :call
                func = args[1]
                types_array = map(args[2:end]) do x
                    t = Sugar.expr_type(m, x)
                    if t == Any
                        error("Found type any in Expr: $expr. Make sure your code is type stable")
                    end
                    t
                end
                types = (types_array...,)
                f = Sugar.resolve_func(m, func)
                func, _was_rewritten = rewrite_for_cudanative(f, types)
                # function arguments have to be rewritten as well!
                args_rewritten = false
                fargs = map(args[2:end]) do x
                    rewr, rewritten = rewrite_expr_cu(m, x)
                    args_rewritten = rewritten || args_rewritten
                    rewr
                end
                was_rewritten = _was_rewritten || args_rewritten || was_rewritten
                return true, Sugar.similar_expr(expr, [func, fargs...])
            end
        end
        false, expr
    end)
    body, was_rewritten
end

"""
Replaces recursively Julia Base functions which are defined as intrinsics in CUDAnative
Returns the resulting function and a bool indicating, wether the function was rewritten.
"""
function rewrite_for_cudanative(f, types)
    fsym = Symbol(f)
    # if defined in CUDAnative, but doesn't match the function passed in here
    # The probability is very high, that this is a cudanative intrinsic
    if isdefined(CUDAnative, fsym) && getfield(CUDAnative, fsym) != f
        cu_f = getfield(CUDAnative, fsym)
        # only if there is a method defined for the argument types
        if !isempty(methods(cu_f, types))
            return cu_f, true # replace it!
        end
    end
    # it's not a CUDAnative intrinsic, so now we need to check it's source for intrinsics
    m = LazyMethod((f, types))
    # if is a Julia intrinsic, stop
    Sugar.isintrinsic(m) && return f, false
    # otherwise go through the source and rewrite function calls recursevely the source!
    expr = Expr(:block, Sugar.get_ast(code_typed, m.signature...)...)
    body, was_rewritten = rewrite_expr_cu(m, expr)
    if was_rewritten
        rewritten_func_expr = Sugar.get_func_expr(m, body, gensym(string("cu_", fsym)))
        eval(rewritten_func_expr), true
    else
        f, false
    end
end
