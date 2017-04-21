export @profile

"""
    @profile

`@profile <expression>` runs your expression, while activating the CUDAnative profiler. This
profiler measures kernel execution times by inserting timing events in the execution stream.

Note that these measurements are slightly inaccurate, including a fixed overhead of calling
into the CUDA API, which in turn causes a overhead that scales in terms of the number and
size of arguments to the kernel.
"""
macro profile(ex)
    quote
        Profile.enabled[] = true
        $(esc(ex))
        Profile.nvprof_enabled[] && Profile.stop_nvprof()
        Profile.enabled[] = false
        Profile.processed[] = false
    end
end


module Profile

using CUDAdrv


## control

const enabled = Ref(false)

const nvprof_enabled = Ref(false)
const nvprof_started = Ref(false)

"""
    init(; nvprof::Bool)

Configures how the CUDAnative profiler behaves. The keyword argument `nvprof` controls
whether the profiler also activates the native CUDA profiler, upon first kernel launch, for
use with `nvprof`.
"""
function init(; nvprof::Bool = false)
    nvprof_enabled[] = nvprof
end

# lazy start the CUDA profiler, to get more compact traces in eg. nvvp
function start_nvprof()
    if !nvprof_started[]
        nvprof_started[] = true
        CUDAdrv.start_profiler()
    end
end

function stop_nvprof()
    if nvprof_started[]
        CUDAdrv.stop_profiler()
        nvprof_started[] = false
    end
end


## instrumentation

struct Launch
    start::CuEvent
    stop::CuEvent

    Launch() = new(CuEvent(), CuEvent())
end
const launches = Vector{Tuple{Symbol,Launch}}()

macro instr_launch(kernel, stream, ex)
    quote
        # tic
        if enabled[]
            lnch = Launch()
            push!(launches, ($(QuoteNode(kernel)), lnch))
            nvprof_enabled[] && start_nvprof()
            record(lnch.start, $(esc(stream)))
        end

        $(esc(ex))

        # toc
        if enabled[]
            record(lnch.stop, $(esc(stream)))
        end
    end
end


## data processing

struct Data
    kernels::Dict{Symbol,Vector{Float64}}
    Data() = new(Dict{Symbol,Vector{Float64}}())
end
const data = Data()

# lazy-process, not to mess with eg. `Base.@elapsed CUDAnative.@profile ...`
const processed = Ref(false)
function process()
    processed[] && return

    kernels = unique(map(t->t[1], launches))
    for kernel in kernels, lnch in map(t->t[2], filter(t->t[1]==kernel, launches))
        synchronize(lnch.stop)
        t = elapsed(lnch.start, lnch.stop)

        push!(get!(data.kernels, kernel, Float64[]), t)
    end

    empty!(launches)
    processed[] = true
end


## reporting

"""
    fetch() -> data

Returns a reference to the internal profiler state. Note that subsequent operations, like
[`clear`](@ref), can affect `data` unless you first make a copy.
"""
function fetch()
    Profile.process()
    return data
end

"""
    clear()

Clear any existing data from the internal buffer.
"""
function clear()
    Profile.process()
    empty!(data.kernels)
end

print(;kwargs...) = print(STDOUT, fetch(); kwargs...)

"""
    print([io::IO = STDOUT,] [data::Vector]; kwargs...)

Prints profiling results to `io` (by default, `STDOUT`). If you do not supply a `data`
argument, the default internal buffer will be used.

The keyword arguments can be any combination of:

 - `format` -- Determines how timings are printed. Possible values, :summary, :csv.
"""
function print(io, data; format=:summary)
    Profile.process()
    any(fmt->fmt == format, [:csv, :summary]) || error("Unknown format")

    # header
    if format == :csv
        println(io, "kernel,time(μs)")
    end

    # data
    for (kernel, times) in data.kernels
        if format == :csv
            for t in times
                println(io, "$kernel,$(1_000_000*t)")
            end
        elseif format == :summary
            println("$kernel: ", round(1_000_000*median(times), 2), "us")
        end
    end

    # footer
    if format == :csv
        println(io, "")
    end
end

end
