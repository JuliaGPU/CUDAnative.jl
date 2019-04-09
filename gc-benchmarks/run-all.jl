using CUDAdrv, CUDAnative

include("utils.jl")

include("arrays.jl")
include("binary-tree.jl")
include("linked-list.jl")
include("matrix.jl")
include("ssa-opt.jl")

println(run_benchmarks())
