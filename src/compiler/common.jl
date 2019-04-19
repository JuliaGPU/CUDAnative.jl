# common functionality

struct CompilerJob
    # core invocation
    f::Core.Function
    tt::DataType
    cap::VersionNumber
    kernel::Bool

    # optional properties
    minthreads::Union{Nothing,CuDim}
    maxthreads::Union{Nothing,CuDim}
    blocks_per_sm::Union{Nothing,Integer}
    maxregs::Union{Nothing,Integer}
    # The name of the 'malloc' function to use when allocating memory.
    # A transform will rewrite all calls to 'malloc' to use this function
    # instead. The 'malloc' signature must be 'void* malloc(size_t)' or
    # compatible.
    malloc::String
    # Indicates whether the GPU GC or the "malloc never free"
    # GC intrinsic lowering strategy is to be used. The former
    # is used when this field is `true`; the latter when it is
    # `false`.
    gc::Bool

    CompilerJob(f, tt, cap, kernel;
                    minthreads=nothing, maxthreads=nothing,
                    blocks_per_sm=nothing, maxregs=nothing,
                    malloc="malloc",gc=false) =
        new(f, tt, cap, kernel, minthreads, maxthreads, blocks_per_sm, maxregs, malloc, gc)
end

# global job reference
# FIXME: thread through `job` everywhere (deadlocks the Julia compiler when doing so with
#        the LLVM passes in CUDAnative)
current_job = nothing


function signature(job::CompilerJob)
    fn = typeof(job.f).name.mt.name
    args = join(job.tt.parameters, ", ")
    return "$fn($(join(job.tt.parameters, ", ")))"
end


struct KernelError <: Exception
    job::CompilerJob
    message::String
    help::Union{Nothing,String}
    bt::StackTraces.StackTrace

    KernelError(job::CompilerJob, message::String, help=nothing;
                bt=StackTraces.StackTrace()) =
        new(job, message, help, bt)
end

function Base.showerror(io::IO, err::KernelError)
    println(io, "GPU compilation of $(signature(err.job)) failed")
    println(io, "KernelError: $(err.message)")
    println(io)
    println(io, something(err.help, "Try inspecting the generated code with any of the @device_code_... macros."))
    Base.show_backtrace(io, err.bt)
end


struct InternalCompilerError <: Exception
    job::CompilerJob
    message::String
    meta::Dict
    InternalCompilerError(job, message; kwargs...) = new(job, message, kwargs)
end

function Base.showerror(io::IO, err::InternalCompilerError)
    println(io, """CUDAnative.jl encountered an unexpected internal compiler error.
                   Please file an issue attaching the following information, including the backtrace,
                   as well as a reproducible example (if possible).""")

    println(io, "\nInternalCompilerError: $(err.message)")

    println(io, "\nCompiler invocation:")
    for field in fieldnames(CompilerJob)
        println(io, " - $field = $(repr(getfield(err.job, field)))")
    end

    if !isempty(err.meta)
        println(io, "\nAdditional information:")
        for (key,val) in err.meta
            println(io, " - $key = $(repr(val))")
        end
    end

    println(io, "\nInstalled packages:")
    for (pkg,ver) in Pkg.installed()
        println(io, " - $pkg = $ver")
    end

    println(io)
    versioninfo(io)
end

macro compiler_assert(ex, job, kwargs...)
    msg = "$ex, at $(__source__.file):$(__source__.line)"
    return :($(esc(ex)) ? $(nothing)
                        : throw(InternalCompilerError($(esc(job)), $msg;
                                                      $(map(esc, kwargs)...)))
            )
end


# maintain our own "global unique" suffix for disambiguating kernels
globalUnique = 0
