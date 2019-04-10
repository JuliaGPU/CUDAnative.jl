using CUDAdrv, CUDAnative, Test

include("utils.jl")

include("arrays.jl")
include("binary-tree.jl")
include("linked-list.jl")
include("matrix.jl")
include("ssa-opt.jl")
include("stream-queries.jl")
include("genetic-algorithm.jl")

results = run_benchmarks()
# Print the results to the terminal.
println(results)

# Also write them to a CSV for further analysis.
open("results.csv", "w") do file
    write(file, "benchmark,nogc,gc,ratio\n")
    for key in sort([k for k in keys(results)])
        runs = results[key]
        median_times = BenchmarkTools.median(runs)
        gc_time = median_times["gc"].time / 1e6
        nogc_time = median_times["nogc"].time / 1e6
        ratio = gc_time / nogc_time
        write(file, "$key,$nogc_time,$gc_time,$ratio\n")
    end
end
