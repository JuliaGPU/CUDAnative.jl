###
# Using Cxx.jl to compile a C++ cuda kernel and then call it from Julia. 
###
using Cxx

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

# Inline cxx doesn't work yet. But hopefully at some point.
# @target ptx function saxpy(a, x, y, out, n)
#   icxx"""
#   size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
#   if (tid < $n) {
#     $out[tid] = $a * $x[tid] + $y[tid];
#   }
#   """
# end

cxx"""
__device__ void saxpy(float a, float *x, float *y, float *out, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = a * x[tid] + y[tid];
  }
}
"""

@target ptx saxpy(a, x, y, out, n) = @cxx saxpy(a, x, y, out, n)

dev = CuDevice(0)
ctx = CuContext(dev)

N = 100
a = 0.5f0
x = rand(Float32, N)
y = rand(Float32, N)

X = CuArray(x)
Y = CuArray(y)
out = CuArray(Float32, (N,))

@cuda (N, 1) saxpy(a, X, Y, out, N)

# Test results
res1 = a .* x .+ y
res2 = Array(out) 

all(res1 .== res2)
