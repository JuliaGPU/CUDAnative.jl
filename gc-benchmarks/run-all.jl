using CUDAdrv, CUDAnative

include("utils.jl")

include("arrays.jl")
include("binary-tree.jl")
include("linked-list.jl")
include("matrix.jl")

println(run_benchmarks())
