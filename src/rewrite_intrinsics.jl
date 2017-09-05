import Sugar
using Sugar: LazyMethod, expr_type, resolve_func, similar_expr, replace_expr, getfunction, isintrinsic


## intrinsic rewriting

const intrinsic_map = Dict{Function,Function}(
    Base.sin    => CUDAnative.sin,
    Base.cos    => CUDAnative.cos
)

# rewrite intrinsics in the source of a LazyMethod
function rewrite_intrinsics(m::LazyMethod, expr)
    changed = false
    body = first(replace_expr(expr) do expr
        if isa(expr, Expr)
            args, head = expr.args, expr.head
            if head == :call
                func = args[1]
                types_array = expr_type.(m, args[2:end])
                f = resolve_func(m, func)
                types = (types_array...,)
                func, call_changed = rewrite_intrinsics(f, types)
                # function arguments have to be rewritten as well!
                args_changed = false
                fargs = map(args[2:end]) do x
                    rewr, changed = rewrite_intrinsics(m, x)
                    args_changed = changed || args_changed
                    rewr
                end
                changed |= call_changed || args_changed
                return true, similar_expr(expr, [func, fargs...])
            end
        end
        false, expr
    end)
    body, changed
end

function rewrite_intrinsics(f::Function, types)
    # rewrite the function itself
    if haskey(intrinsic_map, f)
        return intrinsic_map[f], true
    end

    # get the source and rewrite static parameters
    m = LazyMethod(f, types)
    isintrinsic(m) && return f, false                   # don't rewrite Julia intrinsics
    isa(getfunction(m), DataType) && return f, false    # don't rewrite constructors
    expr = try
        expr = Sugar.sugared(m.signature..., code_typed)
        sparams = Sugar.get_static_parameters(m)
        if !isempty(sparams)
            expr = first(Sugar.replace_expr(expr) do expr
                if isa(expr, Expr) && expr.head == :static_parameter
                    true, sparams[expr.args[1]]
                else
                    false, expr
                end
            end)
        end
        expr
    catch e
        warn(e)
        return f, false
    end

    # rewrite the source
    body, changed = rewrite_intrinsics(m, expr)
    if changed
        changed_func_expr = Sugar.get_func_expr(m, body, gensym(string("cu_", fsym)))
        eval(changed_func_expr), true
    else
        f, false
    end
end
