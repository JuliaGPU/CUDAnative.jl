import Sugar
using Sugar: LazyMethod, expr_type, resolve_func, similar_expr, replace_expr, getfunction, isintrinsic

function rewrite_expr_cu(m::LazyMethod, expr)
    was_rewritten = false
    body = first(replace_expr(expr) do expr
        if isa(expr, Expr)
            args, head = expr.args, expr.head
            if head == :call
                func = args[1]
                types_array = expr_type.(m, args[2:end])
                f = resolve_func(m, func)
                types = (types_array...,)
                func, _was_rewritten = rewrite_for_cudanative(f, types)
                # function arguments have to be rewritten as well!
                args_rewritten = false
                fargs = map(args[2:end]) do x
                    rewr, rewritten = rewrite_expr_cu(m, x)
                    args_rewritten = rewritten || args_rewritten
                    rewr
                end
                was_rewritten = _was_rewritten || args_rewritten || was_rewritten
                return true, similar_expr(expr, [func, fargs...])
            end
        elseif isa(expr, Symbol)
            return true, QuoteNode(expr)
        end
        false, expr
    end)
    body, was_rewritten
end


print_exec(f, args...) = (println(args); f(args...))
"""
Replaces recursively Julia Base functions which are defined as intrinsics in CUDAnative
Returns the resulting function and a bool indicating, wether the function was rewritten.
"""
function rewrite_for_cudanative(f, types)
    fsym = Symbol(f)
    # TODO cover all intrinsics
    if f in (cos, sin, tan, max)
        cu_f = (args...)-> (println(f, args); f(args...))#getfield(CUDAnative, Symbol(f))
        # only if there is a method defined for the argument types
        return cu_f, true # replace it!
        #end
    end
    # it's not a CUDAnative intrinsic, so now we need to check it's source for intrinsics
    m = LazyMethod(f, types)
    # if is a Julia intrinsic, stop
    isintrinsic(m) && return f, false
    # otherwise go through the source and rewrite function calls recursevely the source!
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
        # TODO warn or explicitely filter out errors that are expected?
        # E.g. it can't get the ast for some stuff like Base.cconvert(DataType, x)
        return f, false
    end
    body, was_rewritten = rewrite_expr_cu(m, expr)
    if was_rewritten
        rewritten_func_expr = Sugar.get_func_expr(m, body, gensym(string("cu_", fsym)))
        println(rewritten_func_expr)
        eval(rewritten_func_expr), true
    else
        f, false
    end
end

# struct Test
#     x::Float32
#     Test() = new(sin(1f0))
# end
#
# function test_intrins_recursive(a)
#     tan(Test().x)
# end
# function test_intrinsic_rewrite()
#     a, b = 1f0, 2f0
#     f = cos
#     sin(a) + max(a, b) + f(a) * test_intrins_recursive(a)
# end
#
# f, rewr = rewrite_for_cudanative(test_intrinsic_rewrite, ())
# f()
#
# function cu_Test()
#     return (Test)((convert)(Main.Float32, (test)(1.0f0)))::Test
# end
