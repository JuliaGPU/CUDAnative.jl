# This file implements a high-level generic device-to-host interrupt
# mechanism. This file also contains non-trivial support infrastructure
# that should either be moved to CUDAdrv or exposed by CUDAnative.
# Note that this support infrastructure is not exported, so it remains
# an implementation detail as opposed to a part of CUDAnative's public
# API.

import CUDAdrv: @apicall

export @cuda_interruptible, interrupt, interrupt_or_wait, wait_for_interrupt

# Allocates an array of host memory that is page-locked and accessible
# to the device. Maps the allocation into the CUDA address space.
# Returns a (host array, device buffer) pair. The former can be used by
# the host to access the array, the latter can be used by the device.
function alloc_shared_array(dims::Tuple{Vararg{Int64, N}}, init::T) where {T, N}
    # Allocate memory that is accessible to both the host and the device.
    bytesize = prod(dims) * sizeof(T)
    ptr_ref = Ref{Ptr{Cvoid}}()
    @apicall(
        :cuMemAllocHost,
        (Ptr{Ptr{Cvoid}}, Csize_t),
        ptr_ref, bytesize)

    device_buffer = CUDAdrv.Mem.Buffer(convert(CuPtr{T}, convert(Csize_t, ptr_ref[])), bytesize, CuCurrentContext())

    # Wrap the memory in an array for the host.
    host_array = Base.unsafe_wrap(Array{T, N}, Ptr{T}(ptr_ref[]), dims; own = false)

    # Initialize the array's contents.
    fill!(host_array, init)

    return host_array, device_buffer
end

# Frees an array of host memory.
function free_shared_array(buffer::Mem.Buffer)
    ptr = convert(Ptr{Cvoid}, convert(Csize_t, buffer.ptr))
    @apicall(
        :cuMemFreeHost,
        (Ptr{Cvoid},),
        ptr)
end

# Queries a stream for its status.
function query_stream(stream::CUDAdrv.CuStream = CuDefaultStream())::Cint
    return ccall(
        (:cuStreamQuery, CUDAdrv.libcuda),
        Cint,
        (CUDAdrv.CuStream_t,),
        stream)
end

# Gets a pointer to a global with a particular name. If the global
# does not exist yet, then it is declared in the global memory address
# space.
@generated function get_global_pointer(::Val{global_name}, ::Type{T})::Ptr{T} where {global_name, T}
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

# Gets a pointer to the interrupt region.
@inline function get_interrupt_pointer()::Ptr{UInt32}
    # Compute a pointer to the global in which a pointer to the
    # interrupt state is stored.
    ptr = @cuda_global_ptr("interrupt_pointer", Ptr{UInt32})
    # state the pointer, netting us a pointer to the interrupt
    # region.
    return Base.unsafe_load(ptr)
end

# The interrupt state is a 32-bit unsigned integer that
# can have one of the following values:
#
#   * 0: host is ready to process an interrupt, no interrupt
#        is currently being processed.
#   * 1: device has requested an interrupt, the interrupt
#        has not completed processing yet.
#
const ready = UInt32(0)
const processing = UInt32(1)

"""
    interrupt_or_wait()

Requests an interrupt and waits until the interrupt completes.
If an interrupt is already running, then this function waits
for that interrupt to complete, but does not request an interrupt
of its own. Returns `true` if an interrupt was successfully
requested by this function; otherwise, `false`.
"""
function interrupt_or_wait()::Bool
    state_ptr = get_interrupt_pointer()
    prev_state = atomic_compare_exchange!(state_ptr, ready, processing)
    wait_for_interrupt()
    return prev_state == ready
end

"""
    wait_for_interrupt()

Waits for the current interrupt to finish, if an interrupt is
currently running.
"""
function wait_for_interrupt()
    state_ptr = get_interrupt_pointer()
    while volatile_load(state_ptr) == processing
    end
end

"""
    interrupt()

Repeatedly requests an interrupt until one is requested successfully.
"""
function interrupt()
    while !interrupt_or_wait()
    end
end

# Waits for the current kernel to terminate and handle
# any interrupts that we encounter along the way.
function handle_interrupts(handler::Function, state::Ptr{UInt32}, stream::CuStream = CuDefaultStream())
    while true
        # Sleep to save processing power.
        sleep(0.001)

        # Query the CUDA stream.
        status = query_stream(stream)
        if status == CUDAdrv.SUCCESS.code
            # The kernel has finished running. We're done here.
            return
        elseif status == CUDAdrv.ERROR_NOT_READY.code
            # The kernel is still running. Check if an interrupt
            # needs handling.
            if volatile_load(state) == processing
                # Run the handler.
                handler()
                # Set the interrupt state to 'ready'.
                volatile_store!(state, ready)
            end

            # Continue querying the stream.
        else
            # Whoa. Something both unexpected and unpleasant seems
            # to have happened. Better throw an exception here.
            throw(CuError(status))
        end
    end
end

"""
    @cuda_interruptible [kwargs...] func(args...)

High-level interface for executing code on a GPU with support for interrups.
The `@cuda_interruptible` macro should prefix a call, with `func` a callable function
or object that should return nothing. It will be compiled to a CUDA function upon first
use, and to a certain extent arguments will be converted and anaged automatically using
`cudaconvert`. Finally, a call to `CUDAdrv.cudacall` is performed, scheduling a kernel
launch on the current CUDA context.

Several keyword arguments are supported that influence kernel compilation and execution. For
more information, refer to the documentation of respectively [`cufunction`](@ref) and
[`CUDAnative.Kernel`](@ref).
"""
macro cuda_interruptible(handler, ex...)
    # destructure the `@cuda_interruptible` expression
    if length(ex) > 0 && ex[1].head == :tuple
        error("The tuple argument to @cuda has been replaced by keywords: `@cuda_interruptible handler threads=... fun(args...)`")
    end
    call = ex[end]
    kwargs = ex[1:end-1]

    # destructure the kernel call
    if call.head != :call
        throw(ArgumentError("second argument to @cuda_interruptible should be a function call"))
    end
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    compiler_kwargs, call_kwargs, env_kwargs = CUDAnative.split_kwargs(kwargs)
    vars, var_exprs = CUDAnative.assign_args!(code, args)

    # Find the stream on which the kernel is to be scheduled.
    stream = CuDefaultStream()
    for kwarg in call_kwargs
        key, val = kwarg.args
        if key == :stream
            stream = val
        end
    end

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(code.args,
        quote
            GC.@preserve $(vars...) begin
                # Define a trivial buffer that contains the interrupt state.
                local host_array, device_buffer = alloc_shared_array((1,), ready)

                try
                    # Define a kernel initialization function that sets the
                    # interrupt state pointer.
                    local function interrupt_kernel_init(kernel)
                        try
                            global_handle = CuGlobal{CuPtr{UInt32}}(kernel.mod, "interrupt_pointer")
                            set(global_handle, CuPtr{UInt32}(device_buffer.ptr))
                        catch exception
                            # The interrupt pointer may not have been declared (because it is unused).
                            # In that case, we should do nothing.
                            if !isa(exception, CUDAdrv.CuError) || exception.code != CUDAdrv.ERROR_NOT_FOUND.code
                                rethrow()
                            end
                        end
                    end

                    # Standard kernel setup logic.
                    local kernel_args = CUDAnative.cudaconvert.(($(var_exprs...),))
                    local kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
                    local kernel = CUDAnative.cufunction($(esc(f)), kernel_tt; $(map(esc, compiler_kwargs)...))
                    CUDAnative.prepare_kernel(kernel; init=interrupt_kernel_init, $(map(esc, env_kwargs)...))
                    kernel(kernel_args...; $(map(esc, call_kwargs)...))

                    # Handle interrupts.
                    handle_interrupts($(esc(handler)), pointer(host_array, 1), $(esc(stream)))
                finally
                    free_shared_array(device_buffer)
                end
            end
         end)
    return code
end
