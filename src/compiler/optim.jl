# LLVM IR optimization

function optimize!(job::CompilerJob, mod::LLVM.Module, entry::LLVM.Function; internalize::Bool=true)
    tm = machine(job.cap, triple(mod))

    if job.kernel
        entry = promote_kernel!(job, mod, entry)
    end

    function initialize!(pm)
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)
        if internalize
            # We want to internalize functions so we can optimize
            # them, but we don't really want to internalize globals
            # because doing so may cause multiple copies of the same
            # globals to appear after linking together modules.
            #
            # For example, the runtime library includes GC-related globals.
            # It is imperative that these globals are shared by all modules,
            # but if they are internalized before they are linked then
            # they will actually not be internalized.
            #
            # Also, don't internalize the entry point, for obvious reasons.
            non_internalizable_names = [LLVM.name(entry)]
            for val in globals(mod)
                if isa(val, LLVM.GlobalVariable)
                    push!(non_internalizable_names, LLVM.name(val))
                end
            end
            internalize!(pm, non_internalizable_names)
        end
    end

    global current_job
    current_job = job

    # Julia-specific optimizations
    #
    # NOTE: we need to use multiple distinct pass managers to force pass ordering;
    #       intrinsics should never get lowered before Julia has optimized them.
    if VERSION < v"1.2.0-DEV.375"
        # with older versions of Julia, intrinsics are lowered unconditionally so we need to
        # replace them with GPU-compatible counterparts before anything else. that breaks
        # certain optimizations though: https://github.com/JuliaGPU/CUDAnative.jl/issues/340

        ModulePassManager() do pm
            initialize!(pm)
            add!(pm, FunctionPass("LowerGCFrame", eager_lower_gc_frame!))
            aggressive_dce!(pm) # remove dead uses of ptls
            add!(pm, ModulePass("LowerPTLS", lower_ptls!))
            run!(pm, mod)
        end

        ModulePassManager() do pm
            initialize!(pm)
            ccall(:jl_add_optimization_passes, Cvoid,
                  (LLVM.API.LLVMPassManagerRef, Cint),
                  LLVM.ref(pm), Base.JLOptions().opt_level)
            run!(pm, mod)
        end
    else
        ModulePassManager() do pm
            initialize!(pm)
            ccall(:jl_add_optimization_passes, Cvoid,
                  (LLVM.API.LLVMPassManagerRef, Cint, Cint),
                  LLVM.ref(pm), Base.JLOptions().opt_level, #=lower_intrinsics=# 1)
            run!(pm, mod)
        end

        ModulePassManager() do pm
            initialize!(pm)
            if ctx.gc
                add!(pm, FunctionPass("InsertSafepointsGPUGC", fun -> insert_safepoints_gpugc!(fun, entry)))
                add!(pm, ModulePass("FinalLowerGPUGC", lower_final_gc_intrinsics_gpugc!))
            else
                add!(pm, ModulePass("FinalLowerNoGC", lower_final_gc_intrinsics_nogc!))
            end
            aggressive_dce!(pm) # remove dead uses of ptls
            add!(pm, ModulePass("LowerPTLS", lower_ptls!))

            # the Julia GC lowering pass also has some clean-up that is required
            if VERSION >= v"1.2.0-DEV.531"
                late_lower_gc_frame!(pm)
            end

            run!(pm, mod)
        end
    end

    # PTX-specific optimizations
    ModulePassManager() do pm
        initialize!(pm)

        # NVPTX's target machine info enables runtime unrolling,
        # but Julia's pass sequence only invokes the simple unroller.
        loop_unroll!(pm)
        instruction_combining!(pm)  # clean-up redundancy
        licm!(pm)                   # the inner runtime check might be outer loop invariant

        # the above loop unroll pass might have unrolled regular, non-runtime nested loops.
        # that code still needs to be optimized (arguably, multiple unroll passes should be
        # scheduled by the Julia optimizer). do so here, instead of re-optimizing entirely.
        early_csemem_ssa!(pm) # TODO: gvn instead? see NVPTXTargetMachine.cpp::addEarlyCSEOrGVNPass
        dead_store_elimination!(pm)

        constant_merge!(pm)

        # NOTE: if an optimization is missing, try scheduling an entirely new optimization
        # to see which passes need to be added to the list above
        #     LLVM.clopts("-print-after-all", "-filter-print-funcs=$(LLVM.name(entry))")
        #     ModulePassManager() do pm
        #         add_library_info!(pm, triple(mod))
        #         add_transform_info!(pm, tm)
        #         PassManagerBuilder() do pmb
        #             populate!(pm, pmb)
        #         end
        #         run!(pm, mod)
        #     end

        cfgsimplification!(pm)

        # get rid of the internalized functions; now possible unused
        global_dce!(pm)

        run!(pm, mod)
    end

    # we compile a module containing the entire call graph,
    # so perform some interprocedural optimizations.
    #
    # for some reason, these passes need to be distinct from the regular optimization chain,
    # or certain values (such as the constant arrays used to populare llvm.compiler.user ad
    # part of the LateLowerGCFrame pass) aren't collected properly.
    #
    # these might not always be safe, as Julia's IR metadata isn't designed for IPO.
    ModulePassManager() do pm
        dead_arg_elimination!(pm)   # parent doesn't use return value --> ret void

        run!(pm, mod)
    end

    return entry
end


## kernel-specific optimizations

# promote a function to a kernel
# FIXME: sig vs tt (code_llvm vs cufunction)
function promote_kernel!(job::CompilerJob, mod::LLVM.Module, entry_f::LLVM.Function)
    kernel = wrap_entry!(job, mod, entry_f)

    # property annotations
    # TODO: belongs in irgen? doesn't maxntidx doesn't appear in ptx code?

    annotations = LLVM.Value[kernel]

    ## kernel metadata
    append!(annotations, [MDString("kernel"), ConstantInt(Int32(1), JuliaContext())])

    ## expected CTA sizes
    if job.minthreads != nothing
        bounds = CUDAdrv.CuDim3(job.minthreads)
        for dim in (:x, :y, :z)
            bound = getfield(bounds, dim)
            append!(annotations, [MDString("reqntid$dim"),
                                  ConstantInt(Int32(bound), JuliaContext())])
        end
    end
    if job.maxthreads != nothing
        bounds = CUDAdrv.CuDim3(job.maxthreads)
        for dim in (:x, :y, :z)
            bound = getfield(bounds, dim)
            append!(annotations, [MDString("maxntid$dim"),
                                  ConstantInt(Int32(bound), JuliaContext())])
        end
    end

    if job.blocks_per_sm != nothing
        append!(annotations, [MDString("minctasm"),
                              ConstantInt(Int32(job.blocks_per_sm), JuliaContext())])
    end

    if job.maxregs != nothing
        append!(annotations, [MDString("maxnreg"),
                              ConstantInt(Int32(job.maxregs), JuliaContext())])
    end


    push!(metadata(mod), "nvvm.annotations", MDNode(annotations))


    return kernel
end

function wrapper_type(julia_t::Type, codegen_t::LLVMType)::LLVMType
    if !isbitstype(julia_t)
        # don't pass jl_value_t by value; it's an opaque structure
        return codegen_t
    elseif isa(codegen_t, LLVM.PointerType) && !(julia_t <: Ptr)
        # we didn't specify a pointer, but codegen passes one anyway.
        # make the wrapper accept the underlying value instead.
        return eltype(codegen_t)
    else
        return codegen_t
    end
end

# generate a kernel wrapper to fix & improve argument passing
function wrap_entry!(job::CompilerJob, mod::LLVM.Module, entry_f::LLVM.Function)
    entry_ft = eltype(llvmtype(entry_f)::LLVM.PointerType)::LLVM.FunctionType
    @compiler_assert return_type(entry_ft) == LLVM.VoidType(JuliaContext()) job

    # filter out ghost types, which don't occur in the LLVM function signatures
    sig = Base.signature_type(job.f, job.tt)::Type
    julia_types = Type[]
    for dt::Type in sig.parameters
        if !isghosttype(dt)
            push!(julia_types, dt)
        end
    end

    # generate the wrapper function type & definition
    wrapper_types = LLVM.LLVMType[wrapper_type(julia_t, codegen_t)
                                  for (julia_t, codegen_t)
                                  in zip(julia_types, parameters(entry_ft))]
    wrapper_fn = replace(LLVM.name(entry_f), r"^.+?_"=>"ptxcall_") # change the CC tag
    wrapper_ft = LLVM.FunctionType(LLVM.VoidType(JuliaContext()), wrapper_types)
    wrapper_f = LLVM.Function(mod, wrapper_fn, wrapper_ft)

    # emit IR performing the "conversions"
    let builder = Builder(JuliaContext())
        entry = BasicBlock(wrapper_f, "entry", JuliaContext())
        position!(builder, entry)

        wrapper_args = Vector{LLVM.Value}()

        # perform argument conversions
        codegen_types = parameters(entry_ft)
        wrapper_params = parameters(wrapper_f)
        param_index = 0
        for (julia_t, codegen_t, wrapper_t, wrapper_param) in
            zip(julia_types, codegen_types, wrapper_types, wrapper_params)
            param_index += 1
            if codegen_t != wrapper_t
                # the wrapper argument doesn't match the kernel parameter type.
                # this only happens when codegen wants to pass a pointer.
                @compiler_assert isa(codegen_t, LLVM.PointerType) job
                @compiler_assert eltype(codegen_t) == wrapper_t job

                # copy the argument value to a stack slot, and reference it.
                ptr = alloca!(builder, wrapper_t)
                if LLVM.addrspace(codegen_t) != 0
                    ptr = addrspacecast!(builder, ptr, codegen_t)
                end
                store!(builder, wrapper_param, ptr)
                push!(wrapper_args, ptr)
            else
                push!(wrapper_args, wrapper_param)
                for attr in collect(parameter_attributes(entry_f, param_index))
                    push!(parameter_attributes(wrapper_f, param_index), attr)
                end
            end
        end

        call!(builder, entry_f, wrapper_args)

        ret!(builder)

        dispose(builder)
    end

    # early-inline the original entry function into the wrapper
    push!(function_attributes(entry_f), EnumAttribute("alwaysinline", 0, JuliaContext()))
    linkage!(entry_f, LLVM.API.LLVMInternalLinkage)

    fixup_metadata!(entry_f)
    ModulePassManager() do pm
        always_inliner!(pm)
        verifier!(pm)
        run!(pm, mod)
    end

    return wrapper_f
end

# HACK: get rid of invariant.load and const TBAA metadata on loads from pointer args,
#       since storing to a stack slot violates the semantics of those attributes.
# TODO: can we emit a wrapper that doesn't violate Julia's metadata?
function fixup_metadata!(f::LLVM.Function)
    for param in parameters(f)
        if isa(llvmtype(param), LLVM.PointerType)
            # collect all uses of the pointer
            worklist = Vector{LLVM.Instruction}(user.(collect(uses(param))))
            while !isempty(worklist)
                value = popfirst!(worklist)

                # remove the invariant.load attribute
                md = metadata(value)
                if haskey(md, LLVM.MD_invariant_load)
                    delete!(md, LLVM.MD_invariant_load)
                end
                if haskey(md, LLVM.MD_tbaa)
                    delete!(md, LLVM.MD_tbaa)
                end

                # recurse on the output of some instructions
                if isa(value, LLVM.BitCastInst) ||
                   isa(value, LLVM.GetElementPtrInst) ||
                   isa(value, LLVM.AddrSpaceCastInst)
                    append!(worklist, user.(collect(uses(value))))
                end

                # IMPORTANT NOTE: if we ever want to inline functions at the LLVM level,
                # we need to recurse into call instructions here, and strip metadata from
                # called functions (see CUDAnative.jl#238).
            end
        end
    end
end

# Visits all calls to a particular intrinsic in a given LLVM module.
function visit_calls_to(visit_call::Function, name::AbstractString, mod::LLVM.Module)
    if haskey(functions(mod), name)
        func = functions(mod)[name]

        for use in uses(func)
            call = user(use)::LLVM.CallInst
            visit_call(call, func)
        end
    end
end

# Deletes all calls to a particular intrinsic in a given LLVM module.
# Returns a Boolean that tells if any calls were actually deleted.
function delete_calls_to!(name::AbstractString, mod::LLVM.Module)::Bool
    changed = false
    visit_calls_to(name, mod) do call, _
        unsafe_delete!(LLVM.parent(call), call)
        changed = true
    end
    return changed
end

# lower object allocations to to PTX malloc
#
# this is a PoC implementation that is very simple: allocate, and never free. it also runs
# _before_ Julia's GC lowering passes, so we don't get to use the results of its analyses.
# when we ever implement a more potent GC, we will need those results, but the relevant pass
# is currently very architecture/CPU specific: hard-coded pool sizes, TLS references, etc.
# such IR is hard to clean-up, so we probably will need to have the GC lowering pass emit
# lower-level intrinsics which then can be lowered to architecture-specific code.
function eager_lower_gc_frame!(fun::LLVM.Function)
    job = current_job::CompilerJob
    mod = LLVM.parent(fun)
    changed = false

    # plain alloc
    if haskey(functions(mod), "julia.gc_alloc_obj")
        alloc_obj = functions(mod)["julia.gc_alloc_obj"]
        alloc_obj_ft = eltype(llvmtype(alloc_obj))
        T_prjlvalue = return_type(alloc_obj_ft)
        T_pjlvalue = convert(LLVMType, Any, true)

        for use in uses(alloc_obj)
            call = user(use)::LLVM.CallInst

            # decode the call
            ops = collect(operands(call))
            sz = ops[2]

            # replace with PTX alloc_obj
            let builder = Builder(JuliaContext())
                position!(builder, call)
                ptr = call!(builder, Runtime.get(:gc_pool_alloc), [sz])
                replace_uses!(call, ptr)
                dispose(builder)
            end

            unsafe_delete!(LLVM.parent(call), call)

            changed = true
        end

        @compiler_assert isempty(uses(alloc_obj)) job
    end

    # we don't care about write barriers
    if haskey(functions(mod), "julia.write_barrier")
        barrier = functions(mod)["julia.write_barrier"]

        for use in uses(barrier)
            call = user(use)::LLVM.CallInst
            unsafe_delete!(LLVM.parent(call), call)
            changed = true
        end

        @compiler_assert isempty(uses(barrier)) job
    end
end

# Visits all calls to a particular intrinsic in a given LLVM module
# and redirects those calls to a different function.
# Returns a Boolean that tells if any calls were actually redirected.
function redirect_calls_to!(from::AbstractString, to, mod::LLVM.Module)::Bool
    changed = false
    visit_calls_to(from, mod) do call, _
        args = collect(operands(call))[1:end - 1]
        let builder = Builder(JuliaContext())
            position!(builder, call)
            new_call = call!(builder, to, args)
            replace_uses!(call, new_call)
            unsafe_delete!(LLVM.parent(call), call)
            dispose(builder)
        end
        changed = true
    end
    return changed
end

# Lowers the GC intrinsics produced by the LateLowerGCFrame pass to
# use the "malloc, never free" strategy. These intrinsics are the
# last point at which we can intervene in the pipeline before the
# passes that deal with them become CPU-specific.
function lower_final_gc_intrinsics_nogc!(mod::LLVM.Module)
    changed = false

    # We'll start off with 'julia.gc_alloc_bytes'. This intrinsic allocates
    # store for an object, including headroom, but does not set the object's
    # tag.
    visit_calls_to("julia.gc_alloc_bytes", mod) do call, gc_alloc_bytes
        gc_alloc_bytes_ft = eltype(llvmtype(gc_alloc_bytes))::LLVM.FunctionType
        T_ret = return_type(gc_alloc_bytes_ft)::LLVM.PointerType
        T_bitcast = LLVM.PointerType(T_ret, LLVM.addrspace(T_ret))

        # Decode the call.
        ops = collect(operands(call))
        size = ops[2]

        # We need to reserve a single pointer of headroom for the tag.
        # (LateLowerGCFrame depends on us doing that.)
        headroom = Runtime.tag_size

        # Call the allocation function and bump the resulting pointer
        # so the headroom sits just in front of the returned pointer.
        let builder = Builder(JuliaContext())
            position!(builder, call)
            total_size = add!(builder, size, ConstantInt(Int32(headroom), JuliaContext()))
            ptr = call!(builder, Runtime.get(:gc_pool_alloc), [total_size])
            cast_ptr = bitcast!(builder, ptr, T_bitcast)
            bumped_ptr = gep!(builder, cast_ptr, [ConstantInt(Int32(1), JuliaContext())])
            result_ptr = bitcast!(builder, bumped_ptr, T_ret)
            replace_uses!(call, result_ptr)
            unsafe_delete!(LLVM.parent(call), call)
            dispose(builder)
        end

        changed = true
    end

    # Next up: 'julia.new_gc_frame'. This intrinsic allocates a new GC frame.
    # We'll lower it as an alloca and hope SSA construction and DCE passes
    # get rid of the alloca. This is a reasonable thing to hope for because
    # all intrinsics that may cause the GC frame to escape will be replaced by
    # nops.
    visit_calls_to("julia.new_gc_frame", mod) do call, new_gc_frame
        new_gc_frame_ft = eltype(llvmtype(new_gc_frame))::LLVM.FunctionType
        T_ret = return_type(new_gc_frame_ft)::LLVM.PointerType
        T_alloca = eltype(T_ret)

        # Decode the call.
        ops = collect(operands(call))
        size = ops[1]

        let builder = Builder(JuliaContext())
            position!(builder, call)
            ptr = array_alloca!(builder, T_alloca, size)
            replace_uses!(call, ptr)
            unsafe_delete!(LLVM.parent(call), call)
            dispose(builder)
        end

        changed = true
    end

    # The 'julia.get_gc_frame_slot' is closely related to the previous
    # intrinisc. Specifically, 'julia.get_gc_frame_slot' gets the address of
    # a slot in the GC frame. We can simply turn this intrinsic into a GEP.
    visit_calls_to("julia.get_gc_frame_slot", mod) do call, _
        # Decode the call.
        ops = collect(operands(call))
        frame = ops[1]
        offset = ops[2]

        let builder = Builder(JuliaContext())
            position!(builder, call)
            ptr = gep!(builder, frame, [offset])
            replace_uses!(call, ptr)
            unsafe_delete!(LLVM.parent(call), call)
            dispose(builder)
        end

        changed = true
    end

    # The 'julia.push_gc_frame' registers a GC frame with the GC. We
    # don't have a GC, so we can just delete calls to this intrinsic!
    changed |= delete_calls_to!("julia.push_gc_frame", mod)

    # The 'julia.pop_gc_frame' unregisters a GC frame with the GC, so
    # we can just delete calls to this intrinsic, too.
    changed |= delete_calls_to!("julia.pop_gc_frame", mod)

    # Ditto for 'julia.queue_gc_root'.
    changed |= delete_calls_to!("julia.queue_gc_root", mod)

    return changed
end

"""
lower_final_gc_intrinsics_gpugc!(mod::LLVM.Module)

An LLVM pass that lowers the GC intrinsics produced by the
LateLowerGCFrame pass to use the GPU GC. These intrinsics are the
last point at which we can intervene in the pipeline before the
passes that deal with them become CPU-specific.
"""
function lower_final_gc_intrinsics_gpugc!(mod::LLVM.Module)
    changed = false

    # We'll start off with 'julia.gc_alloc_bytes'. This intrinsic allocates
    # store for an object, including headroom, but does not set the object's
    # tag.
    visit_calls_to("julia.gc_alloc_bytes", mod) do call, gc_alloc_bytes
        gc_alloc_bytes_ft = eltype(llvmtype(gc_alloc_bytes))::LLVM.FunctionType
        T_ret = return_type(gc_alloc_bytes_ft)::LLVM.PointerType
        T_bitcast = LLVM.PointerType(T_ret, LLVM.addrspace(T_ret))

        # Decode the call.
        ops = collect(operands(call))
        size = ops[2]

        # We need to reserve a single pointer of headroom for the tag.
        # (LateLowerGCFrame depends on us doing that.)
        headroom = Runtime.tag_size

        # Call the allocation function and bump the resulting pointer
        # so the headroom sits just in front of the returned pointer.
        let builder = Builder(JuliaContext())
            position!(builder, call)
            total_size = add!(builder, size, ConstantInt(Int32(headroom), JuliaContext()))
            ptr = call!(builder, Runtime.get(:gc_malloc_object), [total_size])
            cast_ptr = bitcast!(builder, ptr, T_bitcast)
            bumped_ptr = gep!(builder, cast_ptr, [ConstantInt(Int32(1), JuliaContext())])
            result_ptr = bitcast!(builder, bumped_ptr, T_ret)
            replace_uses!(call, result_ptr)
            unsafe_delete!(LLVM.parent(call), call)
            dispose(builder)
        end

        changed = true
    end

    # Next up: 'julia.new_gc_frame'. This intrinsic allocates a new GC frame.
    # We actually have a call that implements this intrinsic. Let's use that.
    changed |= redirect_calls_to!("julia.new_gc_frame", Runtime.get(:new_gc_frame), mod)

    # The 'julia.get_gc_frame_slot' is closely related to the previous
    # intrinisc. Specifically, 'julia.get_gc_frame_slot' gets the address of
    # a slot in the GC frame. We can simply turn this intrinsic into a GEP.
    visit_calls_to("julia.get_gc_frame_slot", mod) do call, _
        # Decode the call.
        ops = collect(operands(call))
        frame = ops[1]
        offset = ops[2]

        let builder = Builder(JuliaContext())
            position!(builder, call)
            ptr = gep!(builder, frame, [offset])
            replace_uses!(call, ptr)
            unsafe_delete!(LLVM.parent(call), call)
            dispose(builder)
        end

        changed = true
    end

    # The 'julia.push_gc_frame' registers a GC frame with the GC. We will
    # call a function that does just this.
    changed |= redirect_calls_to!("julia.push_gc_frame", Runtime.get(:push_gc_frame), mod)

    # The 'julia.pop_gc_frame' unregisters a GC frame with the GC. We again
    # have a function in the runtime library.
    changed |= redirect_calls_to!("julia.pop_gc_frame", Runtime.get(:pop_gc_frame), mod)

    # Delete calls to 'julia.queue_gc_root'.
    changed |= delete_calls_to!("julia.queue_gc_root", mod)

    return changed
end

# Tells if a function manages a GC frame.
function has_gc_frame(fun::LLVM.Function)
    for insn in instructions(entry(fun))
        if isa(insn, LLVM.CallInst)
            callee = called_value(insn)
            if isa(callee, LLVM.Function) && LLVM.name(callee) == "julia.new_gc_frame"
                return true
            end
        end
    end
    return false
end

# Tells if an instruction is a call to a non-intrinsic callee.
function is_non_intrinsic_call(instruction::LLVM.Instruction)
    if isa(instruction, LLVM.CallInst)
        callee = called_value(instruction)
        if isa(callee, LLVM.Function)
            callee_name = LLVM.name(callee)
            return !startswith(callee_name, "julia.") && !startswith(callee_name, "llvm.")
        else
            return true
        end
    else
        return false
    end
end

"""
    insert_safepoints_gpugc!(fun::LLVM.Function, entry::LLVM.Function)

An LLVM pass that inserts GC safepoints in such a way that threads
reach a safepoint after a reasonable amount of time.

Moreover, this pass also inserts perma-safepoints after entry point returns.
Perma-safepoints inform the GC that it doesn't need to wait for a warp to
reach a safepoint; inserting them stops the GC from deadlocking.
"""
function insert_safepoints_gpugc!(fun::LLVM.Function, entry::LLVM.Function)
    # Insert a safepoint before every function call, but only for
    # functions that manage a GC frame.
    #
    # TODO: also insert safepoints on loop back-edges? This is what people
    # usually do, but it requires nontrivial IR analyses that the LLVM C
    # API doesn't expose.

    if has_gc_frame(fun)
        safepoint_function = Runtime.get(:gc_safepoint)
        let builder = Builder(JuliaContext())
            for block in blocks(fun)
                for instruction in instructions(block)
                    if is_non_intrinsic_call(instruction)
                        if called_value(instruction) == safepoint_function
                            continue
                        end

                        # Insert a safepoint just before the call.
                        position!(builder, instruction)
                        debuglocation!(builder, instruction)
                        call!(builder, safepoint_function, LLVM.Value[])
                    end
                end
            end
            dispose(builder)
        end
    end

    # Insert perma-safepoints if necessary.
    if fun == entry
        # Looks like we're going to have to insert perma-safepoints.
        # We need to keep in mind that perma-safepoints are per-warp,
        # so we absolutely cannot allow warps to be in a divergent
        # state when a perma-safepoint is set---all bets are off if
        # that happens anyway.
        #
        # To make sure that we don't end up in that situation,
        # we will create a dedicated return block and replace all 'ret'
        # instructions by jumps to that return block.

        # Create the dedicated return block.
        return_block = BasicBlock(fun, "kernel_exit")
        let builder = Builder(JuliaContext())
            position!(builder, return_block)
            call!(builder, Runtime.get(:gc_perma_safepoint), LLVM.Value[])
            ret!(builder)
            dispose(builder)
        end

        # Rewrite return instructions as branches to the return bloc.
        for block in blocks(fun)
            if block == return_block
                # We need to be careful not to trick ourselves into
                # turning the return block's 'ret' into an infinite loop.
                continue
            end
            term = terminator(block)
            if isa(term, LLVM.RetInst)
                unsafe_delete!(block, term)
                let builder = Builder(JuliaContext())
                    position!(builder, block)
                    br!(builder, return_block)
                    dispose(builder)
                end
            end
        end
    end
    return true
end

# lower the `julia.ptls_states` intrinsic by removing it, since it is GPU incompatible.
#
# this assumes and checks that the TLS is unused, which should be the case for most GPU code
# after lowering the GC intrinsics to TLS-less code and having run DCE.
#
# TODO: maybe don't have Julia emit actual uses of the TLS, but use intrinsics instead,
#       making it easier to remove or reimplement that functionality here.
function lower_ptls!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false

    if haskey(functions(mod), "julia.ptls_states")
        ptls_getter = functions(mod)["julia.ptls_states"]

        for use in uses(ptls_getter)
            val = user(use)
            if !isempty(uses(val))
                error("Thread local storage is not implemented")
            end
            unsafe_delete!(LLVM.parent(val), val)
            changed = true
        end

        @compiler_assert isempty(uses(ptls_getter)) job
     end

    return changed
end
