# validation of properties and code

function check_method(job::CompilerJob)
    isa(job.f, Core.Builtin) && throw(KernelError(job, "function is not a generic function"))

    # get the method
    ms = Base.methods(job.f, job.tt)
    isempty(ms)   && throw(KernelError(job, "no method found"))
    length(ms)!=1 && throw(KernelError(job, "no unique matching method"))
    m = first(ms)

    # kernels can't return values
    if job.kernel
        rt = Base.return_types(job.f, job.tt)[1]
        if rt != Nothing
            throw(KernelError(job, "kernel returns a value of type `$rt`",
                """Make sure your kernel function ends in `return`, `return nothing` or `nothing`.
                   If the returned value is of type `Union{}`, your Julia code probably throws an exception.
                   Inspect the code with `@device_code_warntype` for more details."""))
        end
    end

    return
end

function check_invocation(job::CompilerJob, entry::LLVM.Function)
    # make sure any non-isbits arguments are unused
    real_arg_i = 0
    sig = Base.signature_type(job.f, job.tt)::Type
    for (arg_i,dt) in enumerate(sig.parameters)
        isghosttype(dt) && continue
        real_arg_i += 1

        if !isbitstype(dt)
            param = parameters(entry)[real_arg_i]
            if !isempty(uses(param))
                throw(KernelError(job, "passing and using non-bitstype argument",
                    """Argument $arg_i to your kernel function is of type $dt.
                       That type is not isbits, and such arguments are only allowed when they are unused by the kernel."""))
            end
        end
    end

    return
end


## IR validation

const IRError = Tuple{String, StackTraces.StackTrace, Any} # kind, bt, meta

struct InvalidIRError <: Exception
    job::CompilerJob
    errors::Vector{IRError}
end

const RUNTIME_FUNCTION = "call to the Julia runtime"
const UNKNOWN_FUNCTION = "call to an unknown function"
const POINTER_FUNCTION = "call through a literal pointer"

function Base.showerror(io::IO, err::InvalidIRError)
    print(io, "InvalidIRError: compiling $(signature(err.job)) resulted in invalid LLVM IR")
    for (kind, bt, meta) in err.errors
        print(io, "\nReason: unsupported $kind")
        if meta != nothing
            if kind == RUNTIME_FUNCTION || kind == UNKNOWN_FUNCTION || kind == POINTER_FUNCTION
                print(io, " (call to ", meta, ")")
            end
        end
        Base.show_backtrace(io, bt)
    end
    return
end

function check_ir(job, args...)
    errors = check_ir!(job, IRError[], args...)
    unique!(errors)
    if !isempty(errors)
        throw(InvalidIRError(job, errors))
    end

    return
end

function check_ir!(job, errors::Vector{IRError}, mod::LLVM.Module)
    for f in functions(mod)
        check_ir!(job, errors, f)
    end

    return errors
end

function check_ir!(job, errors::Vector{IRError}, f::LLVM.Function)
    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.CallInst)
            check_ir!(job, errors, inst)
        end
    end

    return errors
end

const special_fns = (
    # PTX intrinsics
    "vprintf", "__assertfail", "malloc", "free",
    # libdevice
    "__nvvm_reflect",
    # libcudevrt
    "cudaDeviceSynchronize", "cudaGetParameterBufferV2", "cudaLaunchDeviceV2",
    "cudaCGGetIntrinsicHandle", "cudaCGSynchronize"
)

const libjulia = Ref{Ptr{Cvoid}}(C_NULL)

function check_ir!(job, errors::Vector{IRError}, inst::LLVM.CallInst)
    dest = called_value(inst)
    if isa(dest, LLVM.Function)
        fn = LLVM.name(dest)

        # detect calls to undefined functions
        if isdeclaration(dest) && intrinsic_id(dest) == 0 && !(fn in special_fns) && fn != job.malloc
            # figure out if the function lives in the Julia runtime library
            if libjulia[] == C_NULL
                paths = filter(Libdl.dllist()) do path
                    name = splitdir(path)[2]
                    startswith(name, "libjulia")
                end
                libjulia[] = Libdl.dlopen(first(paths))
            end

            bt = backtrace(inst)
            if Libdl.dlsym_e(libjulia[], fn) != C_NULL
                push!(errors, (RUNTIME_FUNCTION, bt, LLVM.name(dest)))
            else
                push!(errors, (UNKNOWN_FUNCTION, bt, LLVM.name(dest)))
            end
        end
    elseif isa(dest, InlineAsm)
        # let's assume it's valid ASM
    elseif isa(dest, ConstantExpr)
        # detect calls to literal pointers
        # FIXME: can we detect these properly?
        # FIXME: jl_apply_generic and jl_invoke also have such arguments
        if occursin("inttoptr", string(dest))
            # extract the literal pointer
            ptr_arg = first(operands(dest))
            @compiler_assert isa(ptr_arg, ConstantInt) job
            ptr_val = convert(Int, ptr_arg)
            ptr = Ptr{Cvoid}(ptr_val)

            # look it up in the Julia JIT cache
            bt = backtrace(inst)
            frames = ccall(:jl_lookup_code_address, Any, (Ptr{Cvoid}, Cint,), ptr, 0)
            if length(frames) >= 1
                @compiler_assert length(frames) == 1 job frames=frames
                fn, file, line, linfo, fromC, inlined, ip = last(frames)
                push!(errors, (POINTER_FUNCTION, bt, fn))
            else
                fn, file, line, linfo, fromC, inlined, ip = last(frames)
                push!(errors, (POINTER_FUNCTION, bt, nothing))
            end
        end
    end

    return errors
end
