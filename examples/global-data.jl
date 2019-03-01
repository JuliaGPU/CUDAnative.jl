using CUDAdrv, CUDAnative, LLVM, LLVM.Interop
using Test

# This example shows that CUDAnative kernels can include global
# data, which may be set by the host.

# Gets a pointer to a global with a particular name. If the global
# does not exist yet, then it is declared in the global memory address
# space.
@generated function get_global_pointer(::Val{global_name}, ::Type{T}) where {global_name, T}
    T_global = convert(LLVMType, T)
    T_result = convert(LLVMType, Ptr{T})

    # Create a thunk that computes a pointer to the global.
    llvm_f, _ = create_function(T_result)
    mod = LLVM.parent(llvm_f)

    # Figure out if the global has been defined already.
    globalSet = LLVM.globals(mod)
    global_name_string = String(global_name)
    if haskey(globalSet, global_name_string)
        global_var = globalSet[global_name_string]
    else
        # If the global hasn't been defined already, then we'll define
        # it in the global address space, i.e., address space one.
        global_var = GlobalVariable(mod, T_global, global_name_string, 1)
        LLVM.initializer!(global_var, LLVM.null(T_global))
    end

    # Generate IR that computes the global's address.
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        # Cast the global variable's type to the result type.
        result = ptrtoint!(builder, global_var, T_result)
        ret!(builder, result)
    end

    # Call the function.
    call_function(llvm_f, Ptr{T})
end

macro cuda_global_ptr(name, type)
    return :(get_global_pointer(
        $(Val(Symbol(name))),
        $(esc(type))))
end

# Define a kernel that copies the global's value into an array.
function kernel(a::CUDAnative.DevicePtr{Float32})
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    ptr = @cuda_global_ptr("test_global", Float32)
    Base.unsafe_store!(a, Base.unsafe_load(ptr), i)
    return
end

magic = 42.f0

# Define a kernel initialization function that sets the global
# to the magic value.
function kernel_init(kernel)
    global_handle = CuGlobal{Float32}(kernel.mod, "test_global")
    set(global_handle, magic)
end

# Allocate a buffer on the GPU.
len = 12
d_a = Mem.alloc(Float32, len)
ptr = Base.unsafe_convert(CuPtr{Float32}, d_a)

# Run the kernel.
@cuda threads=len init=kernel_init kernel(ptr)

# Test that the buffer has indeed been filled with the magic value.
@test Mem.download(Float32, d_a, len) == repeat([magic], len)
