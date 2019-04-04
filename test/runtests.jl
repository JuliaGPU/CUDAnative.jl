using Test

# development often happens in lockstep with other packages,
# so check-out the master branch of those packages.
using Pkg
if haskey(ENV, "GITLAB_CI")
  Pkg.add([PackageSpec(name = x; rev = "master")
           for x in ["CUDAdrv", "LLVM", "CuArrays"]])
end

using CUDAnative, CUDAdrv
import LLVM

include("util.jl")

@testset "CUDAnative" begin

include("base.jl")
include("pointer.jl")
include("codegen.jl")

if CUDAnative.configured
    @test length(devices()) > 0
    if length(devices()) > 0
        # the API shouldn't have been initialized
        @test CuCurrentContext() == nothing

        device_callbacked = nothing
        device_callback = (dev, ctx) -> begin
            device_callbacked = dev
        end
        push!(CUDAnative.device!_listeners, device_callback)

        # now cause initialization
        Mem.alloc(1)
        @test CuCurrentContext() != nothing
        @test device(CuCurrentContext()) == CuDevice(0)
        @test device_callbacked == CuDevice(0)

        device!(CuDevice(0))
        device!(CuDevice(0)) do
            nothing
        end

        device_reset!()

        # test the device selection functionality
        if length(devices()) > 1
            device!(1) do
                @test device(CuCurrentContext()) == CuDevice(1)
            end
            @test device(CuCurrentContext()) == CuDevice(0)

            device!(1)
            @test device(CuCurrentContext()) == CuDevice(1)
        end

        # pick most recent device (based on compute capability)
        global dev = last(sort(collect(devices()); by=capability))
        @info("Testing using device $(name(dev))")
        device!(dev)

        if capability(dev) < v"2.0"
            @warn("native execution not supported on SM < 2.0")
        else
            include("device/codegen.jl")
            include("device/execution.jl")
            include("device/interrupts.jl")
            include("device/pointer.jl")
            include("device/array.jl")
            include("device/cuda.jl")
            include("device/threading.jl")
            include("device/gc.jl")

            #include("examples.jl")
        end
    end
else
    @warn("CUDAnative.jl has not been configured; skipping on-device tests.")
end

CUDAnative.timings()

end
