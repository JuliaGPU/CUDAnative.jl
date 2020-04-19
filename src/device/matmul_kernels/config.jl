struct Config{
    #= Params =#
    MATMUL_SHAPE,               # MNK, overall shape of the MATMUL operation
    BLOCK_SHAPE,                # MNK, shape of each CTA tile
    WARPS_PER_BLOCK,            # scalar, number of warps per CTA

    MEM_A_WARP,                 # MK, shape of each warp tile during memory operations involving matrix A
    MEM_A_THREAD,               # MK, shape of each thread tile during memory operations involving matrix A

    MEM_B_WARP,                 # KN, shape of each warp tile during memory operations involving matrix B
    MEM_B_THREAD,               # KN, shape of each thread tile during memory operations involving matrix B

    MEM_CD_WARP,                # MN, shape of each warp tile during memory operations involving matrix C or D
    MEM_CD_THREAD,              # MN, shape of each thread tile during memory operations involving matrix C or D

    COMPUTE_WARP,               # MNK, shape of each warp tile during the inner loop computations
    COMPUTE_OP_SHAPE,           # MNK, shape of the operation used in the inner loop

    #= Layouts =#
    GLOBAL_A_LAYOUT,            # layout of the A matrix in global memory
    GLOBAL_B_LAYOUT,            # layout of the B matrix in global memory
    GLOBAL_C_LAYOUT,            # layout of the C matrix in global memory
    GLOBAL_D_LAYOUT,            # layout of the D matrix in global memory

    SHARED_A_LAYOUT,            # layout of the A matrix in shared memory
    SHARED_B_LAYOUT,            # layout of the B matrix in shared memory
    SHARED_C_LAYOUT,            # layout of the C matrix in shared memory
    SHARED_D_LAYOUT,            # layout of the D matrix in shared memory

    #= Operator =#
    OPERATOR,                   # which operator to use in the inner loop
   }
end

@inline function Base.getproperty(conf::Type{Config{MATMUL_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR}}, sym::Symbol) where {MATMUL_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR}
    if sym == :launch_args
        return (threads = WARPS_PER_BLOCK * 32, blocks = (MATMUL_SHAPE.M ÷ BLOCK_SHAPE.M, MATMUL_SHAPE.N ÷ BLOCK_SHAPE.N), shmem = 64 * 1024)
    else
        return getfield(conf, sym)
    end
end

function heuristic_block_shape(shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout)
    # Determining the tile size of each block is a little trickier.
    # We apply the following heuristics:
    # 1) Ideally, the block shape in the M and N dimensions should be square or
    #    nearly-square to maximise data reuse. More specifically, tiling sizes
    #    of the form (M = k, N = k) and (M = 2k, N = k) work well.
    # 2) The tile size should be as large as possible.
    # 3) The size in the M and N dimension is limited by the fact that a tile of
    #    the C (and D) matrix of that size must fit in shared memory.
    # 4) The size in the K dimension is limited by the fact that both a M x K tile
    #    of A and a K x N tile of B must fit in shared memory, at the same time.

    num_bytes_A(M, N, K) = prod(Layout.size(shared_a_layout, (M = M, K = K))) * sizeof(Layout.eltype(shared_a_layout))
    num_bytes_B(M, N, K) = prod(Layout.size(shared_b_layout, (K = K, N = N))) * sizeof(Layout.eltype(shared_b_layout))
    num_bytes_C(M, N, K) = prod(Layout.size(shared_c_layout, (M = M, N = N))) * sizeof(Layout.eltype(shared_c_layout))
    num_bytes_D(M, N, K) = prod(Layout.size(shared_d_layout, (M = M, N = N))) * sizeof(Layout.eltype(shared_d_layout))

    next_MN(M, N, K) = M == N ? (2 * M, N, K) : (M, 2 * N, K)
    next_K(M, N, K) = (M, N, 2 * K)

    cur = 1, 1, 1     # M, N, K
    nxt = next_MN(cur...)

    while (max(num_bytes_C(nxt...), num_bytes_D(nxt...)) <= 64 * 1024)
        cur = nxt
        nxt = next_MN(cur...)
    end

    nxt = next_K(cur...)

    while (num_bytes_A(nxt...) + num_bytes_B(nxt...) <= 64 * 1024)
        cur = nxt
        nxt = next_K(cur...)
    end

    return (M = cur[1], N = cur[2], K = cur[3])
end

# Helper function that returns the logical size of a set of adjacent elements, taking care not
# to make the size larger than the parent tile
function adjacent_elements(num, parent_size)
    p = Tuple(parent_size)
    t = (min(num, p[1]), num ÷ min(num, p[1]))

    return typeof(parent_size)(t)
end

function get_config(; gemm_shape, operator, global_a_layout, global_c_layout, kwargs...)
    params = Dict(kwargs)

    # Use some simple heuristics to get sensible defaults for parameters the user does not specify.

    # Get the global layouts for B & D.
    # Fallback to the layouts of A & C, respectively.
    global_b_layout = get(params, :global_b_layout, global_a_layout)
    global_d_layout = get(params, :global_d_layout, global_c_layout)

    # Get the shared layouts for A, B, C, D.
    # For A & B, add padding to reduce bank conflicts, but preserve 128-bit (16 byte) alignment.
    shared_a_layout = get(params, :shared_a_layout,
                          Layout.Padded{global_a_layout, 16 ÷ sizeof(Layout.eltype(global_a_layout))})
    shared_b_layout = get(params, :shared_b_layout,
                          Layout.Padded{global_b_layout, 16 ÷ sizeof(Layout.eltype(global_b_layout))})
    shared_c_layout = get(params, :shared_c_layout, global_c_layout)
    shared_d_layout = get(params, :shared_d_layout, global_d_layout)

    # 8 warps in a 2 x 4 arrangement usually works well
    warps_per_block = get(params, :warps_per_block, 8)
    op_shape = Operator.shape(operator)
    compute_warp = get(params, :compute_warp,
                       map(*, op_shape, (M = 2, N = 4, K = 1)))

    # Apply heuristic for block shape
    block_shape = get(params, :block_shape,
        heuristic_block_shape(shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout))

    # Heuristics for memory tiling sizes:
    # 1) The tiles should encompass 128 bits (16 bytes) to enable vectorisation.
    # 2) The tiles should be as small as possible (i.e. each thread exactly 128 bits) to enable coalescing.

    mem_a_warp = get(params, :mem_a_warp,
        adjacent_elements(32 * 16 ÷ sizeof(Layout.eltype(global_a_layout)), (M = block_shape.M, K = block_shape.K)))
    mem_b_warp = get(params, :mem_b_warp,
        adjacent_elements(32 * 16 ÷ sizeof(Layout.eltype(global_b_layout)), (K = block_shape.K, N = block_shape.N)))
    mem_cd_warp = get(params, :mem_cd_warp,
        adjacent_elements(32 * 16 ÷ sizeof(Layout.eltype(global_c_layout)), (M = block_shape.M, N = block_shape.N)))

    mem_a_thread = get(params, :mem_a_thread,
        adjacent_elements(16 ÷ sizeof(Layout.eltype(global_a_layout)), (M = block_shape.M, K = block_shape.K)))
    mem_b_thread = get(params, :mem_b_thread,
        adjacent_elements(16 ÷ sizeof(Layout.eltype(global_b_layout)), (K = block_shape.K, N = block_shape.N)))
    mem_cd_thread = get(params, :mem_cd_thread,
        adjacent_elements(16 ÷ sizeof(Layout.eltype(global_c_layout)), (M = block_shape.M, N = block_shape.N)))

    return Config{
        #= Params =#
        gemm_shape,
        block_shape,
        warps_per_block,
        mem_a_warp,
        mem_a_thread,
        mem_b_warp,
        mem_b_thread,
        mem_cd_warp,
        mem_cd_thread,
        compute_warp,
        op_shape,

        #= Layouts =#
        global_a_layout,
        global_b_layout,
        global_c_layout,
        global_d_layout,

        shared_a_layout,
        shared_b_layout,
        shared_c_layout,
        shared_d_layout,

        #= Operators =#
        operator,
    }
end
