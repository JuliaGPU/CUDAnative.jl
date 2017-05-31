###
# Using Cxx.jl to compile a C++ cuda kernel and then call it from Julia. 
###
using Cxx, CUDAdrv, CUDAnative

# Figure out if the CPU is set to the correct value
CI = Cxx.new_clang_instance(false, false, target = "nvptx64-nvdia-cuda", CPU = "sm_35")

# Needs at least LLVM_VER=3.8
addHeaderDir(CI, "/opt/cuda/include", kind = C_System)
cxxinclude(CI, "__clang_cuda_runtime_wrapper.h")

const __current_compiler__ = CI

# Cxx.jl needs to have this function defined and this is fine as long as we don't call it.
cxx"""
extern "C" {
  extern int __cxxjl_personality_v0();
}
"""

# Goal is to make device functions available that can be called from a julia cuda kernel.
cxx"""
#include <curand_kernel.h>
"""

# Cxx doesn't calculate the struct size right now so we have to fill it in manually
typealias curandState_t cxxt"curandState_t"{48}
typealias curandStateSobol32_t cxxt"curandStateSobol32_t"{140}

"""
Initialisation method for curandState_t
"""
curand_init(seed, sequence, offset, state::Ptr{curandState_t}) = @cxx curand_init(seed, sequence, offset, state)

"""
Initialisation method for curandStateSobol32_t
"""
curand_init(direction_vectors, offset, state::Ptr{curandStateSobol32_t}) = @cxx curand_init(direction_vectors, offset, state)
curand_init(direction_vectors, scramble_c, offset, state::Ptr{curandStateSobol32_t}) = @cxx curand_init(direction_vectors, scramble_c, offset, state)

curand(state) = @cxx curand(state)
curand_uniform(::Type{Float32}, state) = @cxx curand_uniform(state)
curand_uniform(::Type{Float64}, state) = @cxx curand_uniform_double(state)
curand_normal(::Type{Float32}, state) = @cxx curand_normal(state)
curand_normal(::Type{Float64}, state) = @cxx curand_normal_double(state)
curand_log_normal(::Type{Float32}, state, mean, stddev) =
    @cxx curand_log_normal(state, mean, stddev)
curand_log_normal(::Type{Float64}, state, mean, stddev) =
    @cxx curand_log_normal_double(state, mean, stddev)
curand_poisson(state, lambda) = @cxx curand_poisson(state, lambda)

@target ptx function fillRandom(out, states)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    # Initialize state
    if i < min(length(out), length(states))
        state = pointer(states, i)
        curand_init(0, i, 0, state)
        @inbounds out[i] = curand_uniform(eltype(out), state) # @inbounds is optional
    end
    return nothing
end

##
# Debug output:
##

# code_warntype(STDOUT, fillRandom, (CuDeviceArray{Float32,1}, CuDeviceArray{curandState_t,1}))
# code_warntype(STDOUT, fillRandom, (CuDeviceArray{Float64,1}, CuDeviceArray{curandState_t,1}))

# code_llvm(STDOUT, fillRandom, (CuDeviceArray{Float32,1}, CuDeviceArray{curandState_t,1}))
# code_llvm(STDOUT, fillRandom, (CuDeviceArray{Float64,1}, CuDeviceArray{curandState_t,1}))

dev = CuDevice(0)
ctx = CuContext(dev)

N = 100
state = CuArray(curandState_t, (N,))
out = CuArray(Float32, (N,))

@cuda (N,1) fillRandom(out, state)

