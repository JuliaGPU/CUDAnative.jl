using CUDAdrv, CUDAnative, CUDAatomics, LLVM, LLVM.Interop
using Test

# This example shows that it is possible to use LLVM's atomic compare
# and exchange instructions from CUDAnative kernels.

# Gets a pointer to a global with a particular name. If the global
# does not exist yet, then it is declared in the global memory address
# space.
@generated function atomic_compare_exchange!(ptr::TPtr, cmp::T, new::T) where {TPtr,T}
    T_ptr = convert(LLVMType, TPtr)
    T_val = convert(LLVMType, T)

    # Create a thunk that performs the compare and exchange.
    llvm_f, _ = create_function(T_val, [T_ptr, T_val, T_val])
    mod = LLVM.parent(llvm_f)

    # Generate IR for the thunk.
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        # Cast the pointer to an actual pointer.
        ptr_val = parameters(llvm_f)[1]
        if !isa(ptr_val, LLVM.PointerType)
            ptr_val = inttoptr!(
                builder,
                ptr_val,
                LLVM.PointerType(T_val))
        end

        # Perform an atomic compare and exchange.
        # TODO: find a way to express the sequential consistency ordering
        # that is less brittle than `UInt32(7)`.
        seq_cst = UInt32(7)
        cmpxchg_val = atomic_cmpxchg!(
            builder,
            ptr_val,
            parameters(llvm_f)[2],
            parameters(llvm_f)[3],
            seq_cst,
            seq_cst,
            false)

        result = extract_value!(builder, cmpxchg_val, 0)
        ret!(builder, result)
    end

    # Call the function.
    call_function(llvm_f, T, Tuple{TPtr, T, T}, :((ptr, cmp, new)))
end

# A store that is implemented using an atomic compare and exchange.
# This is overkill as a store implementation, but it shows that
# atomic compare and exchange works.
function wacky_store!(ptr::CUDAnative.DevicePtr{T}, val::T, index::Integer) where T
    atomic_compare_exchange!(
        ptr + (index - 1) * sizeof(T),
        unsafe_load(ptr, index),
        val)
end

# A kernel that swaps the contents of two buffers using atomic compare
# and exchange instructions.
function vswap(a::CUDAnative.DevicePtr{UInt32}, b::CUDAnative.DevicePtr{UInt32})
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    a_val = unsafe_load(a, i)
    b_val = unsafe_load(b, i)
    wacky_store!(b, a_val, i)
    wacky_store!(a, b_val, i)
    return
end

# Decide on buffer dimensions.
dims = (12,)
len = prod(dims)

# Fill two buffers with random garbage.
a = UInt32.(round.(rand(Float32, dims) * 100))
b = UInt32.(round.(rand(Float32, dims) * 100))

# Allocate buffers on the GPU.
d_a = Mem.alloc(UInt32, len)
Mem.upload!(d_a, a)
a_ptr = Base.unsafe_convert(CuPtr{UInt32}, d_a)
d_b = Mem.alloc(UInt32, len)
Mem.upload!(d_b, b)
b_ptr = Base.unsafe_convert(CuPtr{UInt32}, d_b)

# Run the kernel.
@cuda threads=len vswap(a_ptr, b_ptr)

# Test that the buffers have indeed been swapped.
@test Mem.download(UInt32, d_a, len) == b
@test Mem.download(UInt32, d_b, len) == a
