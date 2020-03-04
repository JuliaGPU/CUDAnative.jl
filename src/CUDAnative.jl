module CUDAnative

using CUDAapi
using CUDAdrv

using LLVM
using LLVM.Interop

using Libdl
using Adapt
using TimerOutputs
using DataStructures


## discovery

const deps = joinpath(@__DIR__, "..", "deps", "deps.jl")
isfile(deps) || error("CUDAnative.jl has not been built, please run Pkg.build(\"CUDAnative\").")
include(deps)

"""
    prefix()

Returns the installation prefix directories of the CUDA toolkit in use.
"""
prefix() = toolkit_dirs

"""
    version()

Returns the version of the CUDA toolkit in use.
"""
version() = toolkit_version

"""
    release()

Returns the CUDA release part of the version as returned by [`version`](@ref).
"""
release() = toolkit_release


## source code includes

# needs to be loaded _before_ the compiler infrastructure, because of generated functions
include("device/tools.jl")
include("device/pointer.jl")
include("device/array.jl")
include("device/cuda.jl")
include("device/llvm.jl")
include("device/runtime.jl")

include("init.jl")
include("compatibility.jl")

include("cupti/CUPTI.jl")
include("nvtx/NVTX.jl")

include("compiler.jl")
include("execution.jl")
include("exceptions.jl")
include("reflection.jl")

include("deprecated.jl")

export CUPTI, NVTX


## initialization

const __initialized__ = Ref(false)
functional() = __initialized__[]

const target_support = Ref{Vector{VersionNumber}}()
const ptx_support = Ref{Vector{VersionNumber}}()

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    silent = parse(Bool, get(ENV, "JULIA_CUDA_SILENT", "false")) || precompiling
    verbose = parse(Bool, get(ENV, "JULIA_CUDA_VERBOSE", "false"))

    # if any dependent GPU package failed, expect it to have logged an error and bail out
    if !CUDAdrv.functional()
        verbose && @warn "CUDAnative.jl did not initialize because CUDAdrv.jl failed to"
        return
    end

    try
        configured || error("CUDAnative.jl has not been successfully built, please run Pkg.build(\"CUDAnative\").")

        # warn about compatibility
        if release() < v"9"
            silent || @warn "CUDAnative.jl only supports CUDA 9.0 or higher (your toolkit provides CUDA $(release()))"
        elseif release() > CUDAdrv.release()
            silent || @warn """You are using CUDA toolkit $(release()) with a driver that only supports up to $(CUDAdrv.release()).
                               It is recommended to upgrade your driver, or switch to automatic installation of CUDA."""
        end

        # warn about missing features
        if libcupti === nothing
            silent || @warn("Your CUDA installation does not provide the CUPTI library, CUDAnative.@code_sass will be unavailable")
        end
        if libnvtx === nothing
            silent || @warn("Your CUDA installation does not provide the NVTX library, CUDAnative.NVTX will be unavailable")
        end

        # determine support
        llvm_support = llvm_compat()
        cuda_support = cuda_compat()
        target_support[] = sort(collect(llvm_support.cap ∩ cuda_support.cap))
        isempty(target_support[]) && error("Your toolchain does not support any device capability")
        ptx_support[] = sort(collect(llvm_support.ptx ∩ cuda_support.ptx))
        isempty(ptx_support[]) && error("Your toolchain does not support any PTX ISA")
        precompiling || @debug "Toolchain with LLVM $(LLVM.version()), CUDA driver $(CUDAdrv.version()) and toolkit $(CUDAnative.version()) supports devices $(verlist(target_support[])); PTX $(verlist(ptx_support[]))"

        __init_compiler__()

        resize!(thread_contexts, Threads.nthreads())
        fill!(thread_contexts, nothing)
        CUDAdrv.initializer(maybe_initialize)

        __initialized__[] = true
    catch ex
        # don't actually fail to keep the package loadable
        if !silent
            if verbose
                @error "CUDAnative.jl failed to initialize" exception=(ex, catch_backtrace())
            else
                @info "CUDAnative.jl failed to initialize, GPU functionality unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)"
            end
        end
    end
end

end
