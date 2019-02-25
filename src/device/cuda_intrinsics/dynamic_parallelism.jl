# See CUDA C Programing Guide Appendix D and
# CUDA Dynamic Parallelism Programing Guid as well as
# https://github.com/nvidia-compiler-sdk/nvvmir-samples/tree/master/device-side-launch

##
# TODO:
# - we need to obtain the symbol to a function so that the linker can replace it
#   with the right address
# - We need to emit relocatable code (although we probably already do this)

const Dim3 = CUDAdrv.CuDim3

@generated function cudaGetParameterBuffer(alignment, size)
    T_int = convert(LLVMType, Int)
    T_sizet = convert(LLVMType, Csize_t)
    T_pint8 = LLVM.PointerType(LLVM.Int8Type(JuliaContext()))
    ￼
    # create function
    ￼param_types = LLVMType[T_sizet, T_sizet]
    llvm_f, _ = create_function(T_int, param_types)
    ￼mod = LLVM.parent(llvm_f)
    ￼
    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)
        ￼
        alignment  = parameters(llvm_f)[1]
        size = parameters(llvm_f)[2]
        ￼
        # invoke cudaGetParameterBuffer and return
        f_typ = LLVM.FunctionType(T_pint8, [T_sizet, T_sizet])
        f = LLVM.Function(mod, "cudaGetParameterBuffer", f_typ)
        LLVM.linkage!(f, LLVM.API.LLVMExternalLinkage)
        ￼
        ptr = call!(builder, f, [alignment, size])
        ptrI = ptrtoint!(builder, ptr, T_int)
        ￼
        ret!(builder, ptrI)
    end
    ￼
    call_function(llvm_f, Ptr{UInt8}, Tuple{Csize_t, Csize_t}, :(alignment, size))
end
    ￼
@generated function cudaGetParameterBufferV2(func::F, gridDim::Dim3, blockDim::Dim3, shmemSize::UInt32) where F
    T_int32 = LLVM.Int32Type(JuliaContext())
    T_int = convert(LLVMType, Int)
    T_dim = convert(LLVMType, Dim3)
    T_pint8 = LLVM.PointerType(LLVM.Int8Type(JuliaContext()))

    # create function
    param_types = LLVMType[T_dim, T_dim, T_int32]
    llvm_f, _ = create_function(T_int, param_types)
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        # How to get pointer to f
        func     = LLVM.PointerNull(T_pint8)
        gridDim  = parameters(llvm_f)[1]
        blockDim = parameters(llvm_f)[2]
        shmem    = parameters(llvm_f)[3]

        # invoke cudaGetParameterBuffer and return
        f_typ = LLVM.FunctionType(T_pint8, [T_pint8, T_dim, T_dim, T_int32])
        f = LLVM.Function(mod, "cudaGetParameterBufferV2", f_typ)
        LLVM.linkage!(f, LLVM.API.LLVMExternalLinkage)

        ptr = call!(builder, f, [func, gridDim, blockDim, shmem])
        ptrI = ptrtoint!(builder, ptr, T_int)

        ret!(builder, ptrI)
    end

    call_function(llvm_f, Ptr{UInt8}, Tuple{Dim3, Dim3, UInt32}, :(gridDim, blockDim, shmemSize))
end

@generated function cudaLaunchDevice(func::F, buf, gridDim, blockDim, shmemSize, stream) where F
    T_int32 = LLVM.Int32Type(JuliaContext())
    T_int = convert(LLVMType, Int)
    T_dim = convert(LLVMType, Dim3)
    T_pint8 = LLVM.PointerType(LLVM.Int8Type(JuliaContext()))

    # create function
    param_types = LLVMType[T_int, T_dim, T_dim, T_int32, T_int]
    llvm_f, _ = create_function(T_int, param_types)
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        # How to get pointer to f
        func     = LLVM.PointerNull(T_pint8)
        buf      = inttoptr!(builder, parameters(llvm_f)[1], T_pint8)
        gridDim  = parameters(llvm_f)[2]
        blockDim = parameters(llvm_f)[3]
        shmem    = parameters(llvm_f)[4]
        stream   = inttoptr!(builder, parameters(llvm_f)[5], T_pint8) # this should be a Ptr{CUstream_t}

        # invoke cudaGetParameterBuffer and return
        f_typ = LLVM.FunctionType(T_int32, [T_pint8, T_dim, T_dim, T_int32, T_pint8])
        f = LLVM.Function(mod, "cudaLaunchDevice", f_typ)
        LLVM.linkage!(f, LLVM.API.LLVMExternalLinkage)

        val = call!(builder, f, [func, gridDim, blockDim, shmem])

        ret!(builder, val)
    end

    call_function(llvm_f, Cint, Tuple{Ptr{UInt8}, Dim3, Dim3, UInt32, Ptr{Void}}, :(buf, gridDim, blockDim, shmemSize, stream))
end

@generated function cudaLaunchDeviceV2(buf, stream)
    T_int32 = LLVM.Int32Type(JuliaContext())
    T_int = convert(LLVMType, Int)
    T_dim = convert(LLVMType, Dim3)
    T_pint8 = LLVM.PointerType(LLVM.Int8Type(JuliaContext()))

    # create function
    param_types = LLVMType[T_int, T_int]
    llvm_f, _ = create_function(T_int, param_types)
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        buf      = inttoptr!(builder, parameters(llvm_f)[1], T_pint8)
        stream   = inttoptr!(builder, parameters(llvm_f)[2], T_pint8) # this should be a Ptr{CUstream_t}

        # invoke cudaGetParameterBuffer and return
        f_typ = LLVM.FunctionType(T_int32, [T_pint8, T_pint8])
        f = LLVM.Function(mod, "cudaLaunchDeviceV2", f_typ)
        LLVM.linkage!(f, LLVM.API.LLVMExternalLinkage)

        val = call!(builder, f, [buf, stream])

        ret!(builder, val)
    end

    call_function(llvm_f, Cint, Tuple{Ptr{UInt8}, Ptr{Void}}, :(buf, stream))
end

@generated function cudaDeviceSynchronize()
    T_int32 = LLVM.Int32Type(JuliaContext())

    # create function
    param_types = LLVMType[]
    llvm_f, _ = create_function(T_int32, param_types, "cudaDeviceSynchronize")
    LLVM.linkage!(llvm_f, LLVM.API.LLVMExternalLinkage)

    call_function(llvm_f, Cint, Tuple{})
end
