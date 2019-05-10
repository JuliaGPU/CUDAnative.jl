using CUDAdrv, CUDAnative, Test

include("utils.jl")

include("array-expansion.jl")
include("array-reduction.jl")
include("arrays.jl")
include("binary-tree.jl")
include("bitvector.jl")
include("linked-list.jl")
include("matrix.jl")
include("ssa-opt.jl")
include("static-arrays.jl")
include("stream-queries.jl")
include("genetic-algorithm.jl")

results = run_benchmarks()
# Print the results to the terminal.
println(results)

# Also write them to a CSV for further analysis.
open("results.csv", "w") do file
    write(file, "benchmark,nogc,gc,gc-shared,nogc-ratio,gc-ratio,gc-shared-ratio\n")
    for key in sort([k for k in keys(results)])
        runs = results[key]
        median_times = BenchmarkTools.median(runs)
        gc_time = median_times["gc"].time / 1e6
        gc_shared_time = median_times["gc-shared"].time / 1e6
        nogc_time = median_times["nogc"].time / 1e6
        gc_ratio = gc_time / nogc_time
        gc_shared_ratio = gc_shared_time / nogc_time
        write(file, "$key,$nogc_time,$gc_time,$gc_shared_time,1,$gc_ratio,$gc_shared_ratio\n")
    end
end
