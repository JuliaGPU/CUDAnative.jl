export @profile

"""
    @profile

`@profile <expression>` runs your expression, while activating the CUDAnative profiler
measuring certain CUDA events like kernel launches. This information is saved in an global,
internal buffer.
"""
macro profile(ex)
    quote
        Profile.enabled[] = true
        $(esc(ex))
        Profile.enabled[] = false
        Profile.nvprof_enabled[] && Profile.stop_nvprof()
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


## state

struct Launch
    start::CuEvent
    stop::CuEvent

    Launch() = new(CuEvent(), CuEvent())
end

const kernel_launches = Dict{Symbol,Vector{Launch}}()

"""
    fetch() -> data

Returns the contents of the internal buffer. This can be used for manual inspection, or to
pass to `print`.
"""
function fetch()
    return [kernel_launches]
end

"""
    clear()

Clear any existing data from the internal buffer.
"""
function clear()
    empty!(kernels)
end


## instrumentation

macro instr_launch(kernel, ex)
    if !haskey(kernel_launches, kernel)
        kernel_launches[kernel] = Launch[]
    end

    quote
        # tic
        if enabled[]
            lnch = Launch()
            push!(kernel_launches[$(QuoteNode(kernel))], lnch)
            nvprof_enabled[] && start_nvprof()
            record(lnch.start)
        end

        $(esc(ex))

        # toc
        if enabled[]
            record(lnch.stop)
        end
    end
end


## reporting

print() = print(STDOUT, fetch())

"""
    print([io::IO = STDOUT,] [data::Vector]; kwargs...)

Prints profiling results to `io` (by default, `STDOUT`). If you do not supply a `data`
argument, the default internal buffer will be used.

The keyword arguments can be any combination of:

 - `format` -- Determines how timings are printed. Possible values, :csv.
"""
function print(io, data, format=:csv)
    (kernel_launches, ) = data
    any(fmt->fmt == format, [:csv]) || error("Unknown format")

    # header
    if format == :csv
        println(io, "kernel,time(μs)")
    end

    # data
    for (kernel, launches) in kernel_launches, lnch in launches
        synchronize(lnch.stop)

        if format == :csv
            println(io, "$kernel,$(1_000_000*elapsed(lnch.start, lnch.stop))")
        end
    end
end

end
