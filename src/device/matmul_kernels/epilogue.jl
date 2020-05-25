module Epilogue

using CUDAnative
using CUDAnative.Tiling
using CUDAnative.MatMul
using GPUifyLoops: @unroll

# ----------------
# Default epilogue
# ----------------

struct Default end

@inline function (ep::Default)(d, shmem_d, transform, conf::Type{MatMul.Config{MATMUL_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR}}) where {MATMUL_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR}
    # Constants
    block_i = (blockIdx().x - 1) * BLOCK_SHAPE.M
    block_j = (blockIdx().y - 1) * BLOCK_SHAPE.N

    warpId = (threadIdx().x - 1) ÷ 32 + 1
    laneId = (threadIdx().x - 1) % 32 + 1

    gemm_sz = Tile(MATMUL_SHAPE)
    block_tile = Tile(BLOCK_SHAPE)

    # Cooperatively store a BLOCK_SHAPE.M x BLOCK_SHAPE.N tile of D from shared to global memory within one threadblock
    @unroll for warp_tile = parallellise(block_tile.MN, Tile(MEM_CD_WARP), warpId, WARPS_PER_BLOCK)
        @unroll for thread_tile = parallellise(warp_tile, Tile(MEM_CD_THREAD), laneId, 32)
            x = Layout.load(SHARED_D_LAYOUT, shmem_d, thread_tile, block_tile.MN.size)
            x = transform(x, thread_tile)
            Layout.store!(GLOBAL_D_LAYOUT, d, x, translate(thread_tile, (M = block_i, N = block_j)), gemm_sz.MN.size)
        end
    end
end

end
