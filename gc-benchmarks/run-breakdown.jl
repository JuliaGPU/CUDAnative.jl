using CUDAdrv, CUDAnative, Test, Statistics, JSON

include("utils-common.jl")

const benchmarks = Dict()
global benchmark_results = Dict()
global current_benchmark = nothing

macro cuda_sync(args...)
    esc(quote
        local heap_size = 10 * MiB
        local local_arena_initial_size = div(heap_size, 10)
        local global_arena_initial_size = heap_size - 8 * local_arena_initial_size
        local gc_config = GCConfiguration(
            local_arena_count=8,
            local_arena_initial_size=local_arena_initial_size,
            global_arena_initial_size=global_arena_initial_size)
        local result = CUDAnative.@cuda gc=true gc_config=gc_config $(args...)
        push!(benchmark_results[current_benchmark], result)
    end)
end

macro cuda_benchmark(name, ex)
    esc(quote
        benchmarks[$name] = (() -> $(ex))
    end)
end

include("array-expansion.jl")
include("array-features.jl")
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

function run_benchmarks()
    cache_dir = mkpath("gc-benchmarks/breakdown-cache")
    global benchmark_results = Dict()
    results = Dict()
    for (k, v) in pairs(benchmarks)
        println(k)
        cache_path = "$cache_dir/$(replace(k, " " => "-")).json"
        if isfile(cache_path)
            results[k] = open(cache_path, "r") do file
                JSON.parse(file)
            end
        else
            # Perform a dry run to ensure that compilations are cached.
            global current_benchmark = k
            benchmark_results[k] = []
            v()

            # Run the benchmarks for real.
            benchmark_results[k] = []
            v()
            while sum(map(x -> x.elapsed_time, benchmark_results[k])) < 90
                v()
            end

            results[k] = [
                Dict(
                    "elapsed-time" => r.elapsed_time,
                    "collection-count" => r.collection_count,
                    "collection-poll-time" => r.collection_poll_time,
                    "collection-time" => r.collection_time)
                for (k, r) in pairs(benchmark_results[k])]

            open(cache_path, "w") do file
                JSON.print(file, results[k])
            end
        end
    end
    return results
end

results = run_benchmarks()
# Write results to a CSV file for further analysis.
open("breakdown.csv", "w") do file
    write(file, "benchmark,collection-poll-ratio,collection-ratio,other-ratio\n")
    all_results = []
    function write_line(key, results)
        if length(all_results) == 0
            all_results = [Float64[] for _ in results]
        end
        write(file, "$key,$(join(results, ','))\n")
        for (l, val) in zip(all_results, results)
            push!(l, val)
        end
    end

    for key in sort(collect(keys(results)))
        runs = results[key]
        total_time = mean(getindex.(runs, "elapsed-time"))
        poll_time = mean(getindex.(runs, "collection-poll-time"))
        collection_time = mean(getindex.(runs, "collection-time"))
        poll_ratio = poll_time / total_time
        collection_ratio = collection_time / total_time
        other_ratio = 1.0 - poll_ratio - collection_ratio
        write_line(key, [poll_time, collection_ratio, other_ratio])
    end
    write_line("mean", mean.(all_results))
end
