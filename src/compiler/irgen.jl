# LLVM IR generation

function module_setup(mod::LLVM.Module)
    triple!(mod, Int === Int64 ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda")

    # add debug info metadata
    push!(metadata(mod), "llvm.module.flags",
         MDNode([ConstantInt(Int32(1), JuliaContext()),    # llvm::Module::Error
                 MDString("Debug Info Version"),
                 ConstantInt(DEBUG_METADATA_VERSION(), JuliaContext())]))
end

# make function names safe for PTX
safe_fn(fn::String) = replace(fn, r"[^A-Za-z0-9_]"=>"_")
safe_fn(f::Core.Function) = safe_fn(String(nameof(f)))
safe_fn(f::LLVM.Function) = safe_fn(LLVM.name(f))

# generate a pseudo-backtrace from a stack of methods being emitted
function backtrace(job::CompilerJob, call_stack::Vector{Core.MethodInstance})
    bt = StackTraces.StackFrame[]
    for method_instance in call_stack
        method = method_instance.def
        frame = StackTraces.StackFrame(method.name, method.file, method.line)
        pushfirst!(bt, frame)
    end
    bt
end

# NOTE: we use an exception to be able to display a stack trace using the logging framework
struct MethodSubstitutionWarning <: Exception
    original::Method
    substitute::Method
end
Base.showerror(io::IO, err::MethodSubstitutionWarning) =
    print(io, "You called $(err.original), maybe you intended to call $(err.substitute) instead?")

function compile_method_instance(job::CompilerJob, method_instance::Core.MethodInstance, world)
    function postprocess(ir)
        # get rid of jfptr wrappers
        for llvmf in functions(ir)
            startswith(LLVM.name(llvmf), "jfptr_") && unsafe_delete!(ir, llvmf)
        end

        return
    end

    # set-up the compiler interface
    last_method_instance = nothing
    call_stack = Vector{Core.MethodInstance}()
    dependencies = MultiDict{Core.MethodInstance,LLVM.Function}()
    function hook_module_setup(ref::Ptr{Cvoid})
        ref = convert(LLVM.API.LLVMModuleRef, ref)
        ir = LLVM.Module(ref)
        module_setup(ir)
    end
    function hook_module_activation(ref::Ptr{Cvoid})
        ref = convert(LLVM.API.LLVMModuleRef, ref)
        ir = LLVM.Module(ref)
        postprocess(ir)

        # find the function that this module defines
        llvmfs = filter(llvmf -> !isdeclaration(llvmf) &&
                                 startswith(LLVM.name(llvmf), "julia_") &&
                                 linkage(llvmf) == LLVM.API.LLVMExternalLinkage,
                        collect(functions(ir)))
        @compiler_assert length(llvmfs) == 1 job
        llvmf = first(llvmfs)

        insert!(dependencies, last_method_instance, llvmf)
    end
    function hook_emit_function(method_instance, code, world)
        skip_verifier = false
        if length(call_stack) >= 1
            caller = last(call_stack)
            skip_verifier = caller.def.name === :overdub
        end

        push!(call_stack, method_instance)

        # check for recursion
        if method_instance in call_stack[1:end-1]
            throw(KernelError(job, "recursion is currently not supported";
                              bt=backtrace(job, call_stack)))
        end

        # if last method on stack is overdub skip the Base check and trust in Cassette
        skip_verifier && return

        # check for Base functions that exist in CUDAnative too
        # FIXME: this might be too coarse
        method = method_instance.def
        if Base.moduleroot(method.module) == Base &&
           isdefined(CUDAnative, method_instance.def.name)
            substitute_function = getfield(CUDAnative, method.name)
            tt = Tuple{method_instance.specTypes.parameters[2:end]...}
            if hasmethod(substitute_function, tt)
                method′ = which(substitute_function, tt)
                if Base.moduleroot(method′.module) == CUDAnative
                    @warn "calls to Base intrinsics might be GPU incompatible" exception=(MethodSubstitutionWarning(method, method′), backtrace(job, call_stack))
                end
            end
        end
    end
    function hook_emitted_function(method, code, world)
        @compiler_assert last(call_stack) == method job
        last_method_instance = pop!(call_stack)
    end
    params = Base.CodegenParams(cached             = false,
                                track_allocations  = false,
                                code_coverage      = false,
                                static_alloc       = false,
                                prefer_specsig     = true,
                                module_setup       = hook_module_setup,
                                module_activation  = hook_module_activation,
                                emit_function      = hook_emit_function,
                                emitted_function   = hook_emitted_function)

    # get the code
    ref = ccall(:jl_get_llvmf_defn, LLVM.API.LLVMValueRef,
                (Any, UInt, Bool, Bool, Base.CodegenParams),
                method_instance, world, #=wrapper=#false, #=optimize=#false, params)
    if ref == C_NULL
        throw(InternalCompilerError(job, "the Julia compiler could not generate LLVM IR"))
    end
    llvmf = LLVM.Function(ref)
    ir = LLVM.parent(llvmf)
    postprocess(ir)

    return llvmf, dependencies
end

function irgen(job::CompilerJob, method_instance::Core.MethodInstance, world)
    entry, dependencies = @timeit to[] "emission" compile_method_instance(job, method_instance, world)
    mod = LLVM.parent(entry)

    # link in dependent modules
    @timeit to[] "linking" begin
        # we disable Julia's compilation cache not to poison it with GPU-specific code.
        # as a result, we might get multiple modules for a single method instance.
        cache = Dict{String,String}()

        for called_method_instance in keys(dependencies)
            llvmfs = dependencies[called_method_instance]

            # link the first module
            llvmf = popfirst!(llvmfs)
            llvmfn = LLVM.name(llvmf)
            link!(mod, LLVM.parent(llvmf))

            # process subsequent duplicate modules
            for dup_llvmf in llvmfs
                if Base.JLOptions().debug_level >= 2
                    # link them too, to ensure accurate backtrace reconstruction
                    link!(mod, LLVM.parent(dup_llvmf))
                else
                    # don't link them, but note the called function name in a cache
                    dup_llvmfn = LLVM.name(dup_llvmf)
                    cache[dup_llvmfn] = llvmfn
                end
            end
        end

        # resolve function declarations with cached entries
        for llvmf in filter(isdeclaration, collect(functions(mod)))
            llvmfn = LLVM.name(llvmf)
            if haskey(cache, llvmfn)
                def_llvmfn = cache[llvmfn]
                replace_uses!(llvmf, functions(mod)[def_llvmfn])

                @compiler_assert isempty(uses(llvmf)) job
                unsafe_delete!(LLVM.parent(llvmf), llvmf)
            end
        end
    end

    # clean up incompatibilities
    @timeit to[] "clean-up" for llvmf in functions(mod)
        llvmfn = LLVM.name(llvmf)

        # only occurs in debug builds
        delete!(function_attributes(llvmf), EnumAttribute("sspstrong", 0, JuliaContext()))

        # rename functions
        if !isdeclaration(llvmf)
            # Julia disambiguates local functions by prefixing with `#\d#`.
            # since we don't use a global function namespace, get rid of those tags.
            if occursin(r"^julia_#\d+#", llvmfn)
                llvmfn′ = replace(llvmfn, r"#\d+#"=>"")
                if !haskey(functions(mod), llvmfn′)
                    LLVM.name!(llvmf, llvmfn′)
                    llvmfn = llvmfn′
                end
            end

            # anonymous functions are just named `#\d`, make that somewhat more readable
            m = match(r"_#(\d+)_", llvmfn)
            if m !== nothing
                llvmfn′ = replace(llvmfn, m.match=>"_anonymous$(m.captures[1])_")
                LLVM.name!(llvmf, llvmfn′)
                llvmfn = llvmfn′
            end

            # finally, make function names safe for ptxas
            # (LLVM should to do this, but fails, see eg. D17738 and D19126)
            llvmfn′ = safe_fn(llvmfn)
            if llvmfn != llvmfn′
                LLVM.name!(llvmf, llvmfn′)
                llvmfn = llvmfn′
            end
        end
    end

    # rename the entry point
    if job.name !== nothing
        llvmfn = safe_fn(string("julia_", job.name))
    else
        llvmfn = replace(LLVM.name(entry), r"_\d+$"=>"")
    end
    ## append a global unique counter
    global globalUnique
    globalUnique += 1
    llvmfn *= "_$globalUnique"
    LLVM.name!(entry, llvmfn)

    # minimal required optimization
    @timeit to[] "rewrite" ModulePassManager() do pm
        global current_job
        current_job = job

        linkage!(entry, LLVM.API.LLVMExternalLinkage)
        internalize!(pm, [LLVM.name(entry)])

        add!(pm, ModulePass("LowerThrow", lower_throw!))
        add!(pm, FunctionPass("HideUnreachable", hide_unreachable!))
        add!(pm, ModulePass("HideTrap", hide_trap!))
        always_inliner!(pm)
        run!(pm, mod)
    end

    return mod, entry
end

# this pass lowers `jl_throw` and friends to GPU-compatible exceptions.
# this isn't strictly necessary, but has a couple of advantages:
# - we can kill off unused exception arguments that otherwise would allocate or invoke
# - we can fake debug information (lacking a stack unwinder)
#
# once we have thorough inference (ie. discarding `@nospecialize` and thus supporting
# exception arguments) and proper debug info to unwind the stack, this pass can go.
function lower_throw!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit to[] "lower throw" begin

    throw_functions = Dict{String,String}(
        "jl_throw"                      => "exception",
        "jl_error"                      => "error",
        "jl_too_few_args"               => "too few arguments exception",
        "jl_too_many_args"              => "too many arguments exception",
        "jl_type_error_rt"              => "type error",
        "jl_undefined_var_error"        => "undefined variable error",
        "jl_bounds_error"               => "bounds error",
        "jl_bounds_error_v"             => "bounds error",
        "jl_bounds_error_int"           => "bounds error",
        "jl_bounds_error_tuple_int"     => "bounds error",
        "jl_bounds_error_unboxed_int"   => "bounds error",
        "jl_bounds_error_ints"          => "bounds error",
        "jl_eof_error"                  => "EOF error"
    )

    for (fn, name) in throw_functions
        if haskey(functions(mod), fn)
            f = functions(mod)[fn]

            for use in uses(f)
                call = user(use)::LLVM.CallInst

                # replace the throw with a PTX-compatible exception
                let builder = Builder(JuliaContext())
                    position!(builder, call)
                    emit_exception!(builder, name, call)
                    dispose(builder)
                end

                # remove the call
                call_args = collect(operands(call))[1:end-1] # last arg is function itself
                unsafe_delete!(LLVM.parent(call), call)

                # HACK: kill the exceptions' unused arguments
                for arg in call_args
                    # peek through casts
                    if isa(arg, LLVM.AddrSpaceCastInst)
                        cast = arg
                        arg = first(operands(cast))
                        isempty(uses(cast)) && unsafe_delete!(LLVM.parent(cast), cast)
                    end

                    if isa(arg, LLVM.Instruction) && isempty(uses(arg))
                        unsafe_delete!(LLVM.parent(arg), arg)
                    end
                end

                changed = true
            end

            @compiler_assert isempty(uses(f)) job
         end
     end

    end
    return changed
end

# report an exception in a GPU-compatible manner
#
# the exact behavior depends on the debug level. in all cases, a `trap` will be emitted, On
# debug level 1, the exception name will be printed, and on debug level 2 the individual
# stack frames (as recovered from the LLVM debug information) will be printed as well.
function emit_exception!(builder, name, inst)
    bb = position(builder)
    fun = LLVM.parent(bb)
    mod = LLVM.parent(fun)

    # report the exception
    if Base.JLOptions().debug_level >= 1
        name = globalstring_ptr!(builder, name, "exception")
        if Base.JLOptions().debug_level == 1
            call!(builder, Runtime.get(:report_exception), [name])
        else
            call!(builder, Runtime.get(:report_exception_name), [name])
        end
    end

    # report each frame
    if Base.JLOptions().debug_level >= 2
        rt = Runtime.get(:report_exception_frame)
        bt = backtrace(inst)
        for (i,frame) in enumerate(bt)
            idx = ConstantInt(rt.llvm_types[1], i)
            func = globalstring_ptr!(builder, String(frame.func), "di_func")
            file = globalstring_ptr!(builder, String(frame.file), "di_file")
            line = ConstantInt(rt.llvm_types[4], frame.line)
            call!(builder, rt, [idx, func, file, line])
        end
    end

    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(JuliaContext())))
    end
    call!(builder, trap)
end

# HACK: this pass removes `unreachable` information from LLVM
#
# `ptxas` is buggy and cannot deal with thread-divergent control flow in the presence of
# shared memory (see JuliaGPU/CUDAnative.jl#4). avoid that by rewriting control flow to fall
# through any other block. this is semantically invalid, but the code is unreachable anyhow
# (and we expect it to be preceded by eg. a noreturn function, or a trap).
#
# TODO: can LLVM do this with structured CFGs? It seems to have some support, but seemingly
#       only to prevent introducing non-structureness during optimization (ie. the front-end
#       is still responsible for generating structured control flow).
function hide_unreachable!(fun::LLVM.Function)
    job = current_job::CompilerJob
    changed = false
    @timeit to[] "hide unreachable" begin

    # remove `noreturn` attributes
    #
    # when calling a `noreturn` function, LLVM places an `unreachable` after the call.
    # this leads to an early `ret` from the function.
    attrs = function_attributes(fun)
    delete!(attrs, EnumAttribute("noreturn", 0, JuliaContext()))

    # build a map of basic block predecessors
    predecessors = Dict(bb => Set{LLVM.BasicBlock}() for bb in blocks(fun))
    @timeit to[] "predecessors" for bb in blocks(fun)
        insts = instructions(bb)
        if !isempty(insts)
            inst = last(insts)
            if isterminator(inst)
                for bb′ in successors(inst)
                    push!(predecessors[bb′], bb)
                end
            end
        end
    end

    # scan for unreachable terminators and alternative successors
    worklist = Pair{LLVM.BasicBlock, Union{Nothing,LLVM.BasicBlock}}[]
    @timeit to[] "find" for bb in blocks(fun)
        unreachable = terminator(bb)
        if isa(unreachable, LLVM.UnreachableInst)
            unsafe_delete!(bb, unreachable)
            changed = true

            try
                terminator(bb)
                # the basic-block is still terminated properly, nothing to do
                # (this can happen with `ret; unreachable`)
                # TODO: `unreachable; unreachable`
            catch ex
                isa(ex, UndefRefError) || rethrow(ex)
                let builder = Builder(JuliaContext())
                    position!(builder, bb)

                    # find the strict predecessors to this block
                    preds = collect(predecessors[bb])

                    # find a fallthrough block: recursively look at predecessors
                    # and find a successor that branches to any other block
                    fallthrough = nothing
                    while !isempty(preds)
                        # find an alternative successor
                        for pred in preds, succ in successors(terminator(pred))
                            if succ != bb
                                fallthrough = succ
                                break
                            end
                        end
                        fallthrough === nothing || break

                        # recurse upwards
                        old_preds = copy(preds)
                        empty!(preds)
                        for pred in old_preds
                            append!(preds, predecessors[pred])
                        end
                    end
                    push!(worklist, bb => fallthrough)

                    dispose(builder)
                end
            end
        end
    end

    # apply the pending terminator rewrites
    @timeit to[] "replace" if !isempty(worklist)
        let builder = Builder(JuliaContext())
            for (bb, fallthrough) in worklist
                position!(builder, bb)
                if fallthrough !== nothing
                    br!(builder, fallthrough)
                else
                    # couldn't find any other successor. this happens with functions
                    # that only contain a single block, or when the block is dead.
                    ft = eltype(llvmtype(fun))
                    if return_type(ft) == LLVM.VoidType(JuliaContext())
                        # even though returning can lead to invalid control flow,
                        # it mostly happens with functions that just throw,
                        # and leaving the unreachable there would make the optimizer
                        # place another after the call.
                        ret!(builder)
                    else
                        unreachable!(builder)
                    end
                end
            end
        end
    end

    end
    return changed
end

# HACK: this pass removes calls to `trap` and replaces them with inline assembly
#
# if LLVM knows we're trapping, code is marked `unreachable` (see `hide_unreachable!`).
function hide_trap!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit to[] "hide trap" begin

    # inline assembly to exit a thread, hiding control flow from LLVM
    exit_ft = LLVM.FunctionType(LLVM.VoidType(JuliaContext()))
    exit = InlineAsm(exit_ft, "trap;", "", true)

    if haskey(functions(mod), "llvm.trap")
        trap = functions(mod)["llvm.trap"]

        for use in uses(trap)
            val = user(use)
            if isa(val, LLVM.CallInst)
                let builder = Builder(JuliaContext())
                    position!(builder, val)
                    call!(builder, exit)
                    dispose(builder)
                end
                unsafe_delete!(LLVM.parent(val), val)
                changed = true
            end
        end
    end

    end
    return changed
end
