# C. Cooperative Groups

export this_grid, sync_grid,
       this_multi_grid, sync_multi_grid

@enum cudaCGScope::Cuint begin
     CGScopeInvalid   = 0  # Invalid cooperative group scope
     CGScopeGrid           # Scope represented by a grid_group
     CGScopeMultiGrid      # Scope represented by a multi_grid_group
end


# Intrinsic to get a handle for both grid groups and multi-grid groups:
if VERSION >= v"1.2.0-DEV.512"
    @inline cg_intrinsic_handle(cooperative_scope::cudaCGScope) =
        ccall("extern cudaCGGetIntrinsicHandle", llvmcall, Culonglong, (cudaCGScope,), cooperative_scope)
else
    @eval @inline cg_intrinsic_handle(cooperative_scope::cudaCGScope) = Base.llvmcall(
        $("declare i64 @cudaCGGetIntrinsicHandle(i32)",
          "%rv = call i64 @cudaCGGetIntrinsicHandle(i32 %0)
           ret i64 %rv"), Culonglong,
        Tuple{cudaCGScope}, cooperative_scope)
end

"""
    this_grid()

Returns a `grid_handle` of the grid group this thread belongs to. Only available if a
cooperative kernel is launched.
"""
this_grid() = cg_intrinsic_handle(CGScopeGrid)

"""
    this_multi_grid()

Returns a `multi_grid_handle` of the multi-grid group this thread belongs to. Only available
if a cooperative kernel is launched.
"""
this_multi_grid() = cg_intrinsic_handle(CGScopeMultiGrid)


# Intrinsic to synchronise both grid groups and multi-grid groups:
if VERSION >= v"1.2.0-DEV.512"
    @inline cg_sync_handle(grid_handle::Culonglong) =
        ccall("extern cudaCGSynchronize", llvmcall, cudaError_t,
              (Culonglong, Cuint), grid_handle, UInt32(0))
else
    @eval @inline cg_sync_handle(grid_handle::Culonglong) = Base.llvmcall(
        $("declare i32 @cudaCGSynchronize(i64, i32)",
          "%rv = call i32 @cudaCGSynchronize(i64 %0, i32 0)
           ret i32 %rv"), cudaError_t,
        Tuple{Culonglong}, grid_handle)
end

"""
    sync_grid(grid_handle::Culonglong)

Waits until all threads in all blocks in the grid `grid_handle` have reached this point and
all global memory accesses made by these threads prior to `sync_grid()` are visible to all
threads in the grid. A 32-bit integer `cudaError_t` is returned.
"""
sync_grid(grid_handle::Culonglong) = cg_sync_handle(grid_handle)

"""
    sync_multi_grid(multi_grid_handle::Culonglong)

Waits until all threads in all blocks in the multi-grid `multi_grid_handle` have reached this
point and all global memory accesses made by these threads prior to `sync_multi_grid()` are
visible to all threads in the grid. A 32-bit integer `cudaError_t` is returned.
"""
sync_multi_grid(multi_grid_handle::Culonglong) = cg_sync_handle(multi_grid_handle)
