__precompile__()

module CUDAnative

using LLVM
using CUDAdrv
import CUDAdrv: debug, DEBUG, trace, TRACE

const ext = joinpath(@__DIR__, "..", "deps", "ext.jl")
isfile(ext) || error("Unable to load $ext\n\nPlease run Pkg.build(\"CUDAnative\") and restart Julia.")
include(ext)

include("jit.jl")
include("profile.jl")
include("device/array.jl")
include("device/intrinsics.jl") # these files contain generated functions,
include("execution.jl")         # so should get loaded quite late (JuliaLang/julia#19942)
include("reflection.jl")

function __init__()
    if CUDAdrv.version() != cuda_version ||
        LLVM.version() != llvm_version ||
        VersionNumber(Base.libllvm_version) != julia_llvm_version
        error("Your set-up has changed. Please re-run Pkg.build(\"CUDAnative\") and restart Julia.")
    end
end

end
