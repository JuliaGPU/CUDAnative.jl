################################################################################
# WMMA FRAGMENT
################################################################################

export wmma_row_major, wmma_col_major, wmma_unspecified

abstract type wmma_fragment_layout end
struct wmma_row_major <: wmma_fragment_layout end
struct wmma_col_major <: wmma_fragment_layout end
struct wmma_unspecified <: wmma_fragment_layout end


export wmma_matrix_a, wmma_matrix_b, wmma_accumulator

abstract type wmma_fragment_use end
struct wmma_matrix_a <: wmma_fragment_use end
struct wmma_matrix_b <: wmma_fragment_use end
struct wmma_accumulator <: wmma_fragment_use end


export wmma_fragment

struct wmma_fragment{M, N, K, FS, T, L <: wmma_fragment_layout, U <: wmma_fragment_use}
    x::NTuple{FS, T}
end

################################################################################
# WMMA CONFIGURATION
################################################################################

export wmma_config
struct wmma_config{M, N, K} end

################################################################################
# WMMA LOAD
################################################################################

export wmma_load_a, wmma_load_b, wmma_load_c

function wmma_load_a(addr::DevicePtr{Float16, AS.Global},
                     stride::Number,
                     layout::Type{wmma_col_major},
                     config::Type{wmma_config{16, 16, 16}})
    x = llvm_wmma_load_a_col_m16n16k16_stride_f16(addr, stride)
    return wmma_fragment{16, 16, 16, 8, NTuple{2, VecElement{Float16}}, wmma_col_major, wmma_matrix_a}(x)
end

function wmma_load_b(addr::DevicePtr{Float16, AS.Global},
                     stride::Number,
                     layout::Type{wmma_col_major},
                     config::Type{wmma_config{16, 16, 16}})
    x = llvm_wmma_load_b_col_m16n16k16_stride_f16(addr, stride)
    return wmma_fragment{16, 16, 16, 8, NTuple{2, VecElement{Float16}}, wmma_col_major, wmma_matrix_b}(x)
end

function wmma_load_c(addr::DevicePtr{Float16, AS.Global},
                     stride::Number,
                     layout::Type{wmma_col_major},
                     config::Type{wmma_config{16, 16, 16}})
    x = llvm_wmma_load_c_col_m16n16k16_stride_f16(addr, stride)
    return wmma_fragment{16, 16, 16, 4, NTuple{2, VecElement{Float16}}, wmma_unspecified, wmma_accumulator}(x)
end

################################################################################
# WMMA MMA
################################################################################

export wmma_mma

function wmma_mma(a::wmma_fragment{16, 16, 16, 8, NTuple{2, VecElement{Float16}}, wmma_col_major, wmma_matrix_a},
                  b::wmma_fragment{16, 16, 16, 8, NTuple{2, VecElement{Float16}}, wmma_col_major, wmma_matrix_b},
                  c::wmma_fragment{16, 16, 16, 4, NTuple{2, VecElement{Float16}}, wmma_unspecified, wmma_accumulator})
    x = llvm_wmma_mma_col_col_m16n16k16_f16_f16(a.x, b.x, c.x)
    return wmma_fragment{16, 16, 16, 4, NTuple{2, VecElement{Float16}}, wmma_unspecified, wmma_accumulator}(x)
end

################################################################################
# WMMA STORE
################################################################################

export wmma_store_d

function wmma_store_d(addr::DevicePtr{Float16, AS.Global},
                      d::wmma_fragment{16, 16, 16, 4, NTuple{2, VecElement{Float16}}, wmma_unspecified, wmma_accumulator},
                      stride::Number,
                      layout::Type{wmma_col_major},
                      config::Type{wmma_config{16, 16, 16}})
    llvm_wmma_store_d_col_m16n16k16_stride_f16(addr, d.x, stride)
    return nothing
end

################################################################################
# WMMA FILL FRAGMENT
################################################################################

# TODO
