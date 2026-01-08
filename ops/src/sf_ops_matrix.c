#include <sionflow/ops/sf_ops_core.h>
#include "sf_kernel_utils.h"
#include <sionflow/isa/sf_opcodes.h>
#include <sionflow/base/sf_math.h>
#include "sf_ops_internal.h"
#include <string.h>
#include <math.h>

void op_MATMUL(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    const uint8_t a_ndim = ctx->reg_ndims[inst->src1_idx];
    const uint8_t b_ndim = ctx->reg_ndims[inst->src2_idx];
    const uint8_t d_ndim = ctx->reg_ndims[inst->dest_idx];

    const int32_t M = ctx->reg_shapes[inst->src1_idx][a_ndim - 2];
    const int32_t K = ctx->reg_shapes[inst->src1_idx][a_ndim - 1];
    const int32_t N = ctx->reg_shapes[inst->src2_idx][b_ndim - 1];

    u8* base_a = (u8*)ctx->reg_ptrs[inst->src1_idx];
    u8* base_b = (u8*)ctx->reg_ptrs[inst->src2_idx];
    u8* base_c = (u8*)ctx->reg_ptrs[inst->dest_idx];

    const i32 st_batch_a = SF_GET_STRIDE_S1(inst);
    const i32 st_batch_b = SF_GET_STRIDE_S2(inst);
    const i32 st_batch_c = SF_GET_STRIDE_D(inst);

    // Manual matrix ops still use element-strides, so we use pre-calculated task strides
    const int last_dim = (ctx->ndim > 0) ? (int)ctx->ndim - 1 : 0;
    const int32_t stride_ka = ctx->reg_strides[inst->src1_idx][last_dim];
    
    // For internal matrix indexing, we assume contiguous layout for the matrix part
    const int32_t stride_ra = K * stride_ka;
    const int32_t stride_kb = N * stride_ka; // This assumes B is also contiguous
    const int32_t stride_cb = stride_ka;
    const int32_t stride_cc = stride_ka;
    const int32_t stride_rc = N * stride_ka;

    const size_t batch_size = ctx->batch_size;

    for (size_t b = 0; b < batch_size; ++b) {
        u8* curr_a = base_a + b * st_batch_a;
        u8* curr_b = base_b + b * st_batch_b;
        u8* curr_c = base_c + b * st_batch_c;

        for (int32_t r = 0; r < M; r++) {
            for (int32_t c = 0; c < N; c++) { 
                float sum = 0.0f; 
                u8* pa = curr_a + r * stride_ra;
                u8* pb = curr_b + c * stride_cb;
                for (int32_t k = 0; k < K; k++) {
                    sum += (*(f32*)pa) * (*(f32*)pb);
                    pa += stride_ka;
                    pb += stride_kb;
                }
                *(f32*)(curr_c + r * stride_rc + c * stride_cc) = sum; 
            }
        }
    }
}

void op_TRANSPOSE(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    uint8_t a_ndim = ctx->reg_ndims[inst->src1_idx];
    uint8_t d_ndim = ctx->reg_ndims[inst->dest_idx];
    const int32_t* a_shape = ctx->reg_shapes[inst->src1_idx];
    
    u8* base_a = (u8*)ctx->reg_ptrs[inst->src1_idx];
    u8* base_d = (u8*)ctx->reg_ptrs[inst->dest_idx];

    const i32 st_batch_a = SF_GET_STRIDE_S1(inst);
    const i32 st_batch_d = SF_GET_STRIDE_D(inst);

    // Simplified for 2D (or last 2 dims)
    int rows = a_shape[a_ndim - 2];
    int cols = a_shape[a_ndim - 1];
    
    // We assume contiguous for matrix part
    int s1 = 4; // F32
    int s0 = cols * s1;
    int d1 = 4;
    int d0 = rows * d1;

    for (size_t b = 0; b < ctx->batch_size; ++b) {
        u8* sa = base_a + b * st_batch_a;
        u8* dd = base_d + b * st_batch_d;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                *(f32*)(dd + c * d0 + r * d1) = *(f32*)(sa + r * s0 + c * s1);
            }
        }
    }
}

void op_INVERSE(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    uint8_t a_ndim = ctx->reg_ndims[inst->src1_idx];
    const int32_t* a_shape = ctx->reg_shapes[inst->src1_idx];
    
    u8* base_a = (u8*)ctx->reg_ptrs[inst->src1_idx];
    u8* base_d = (u8*)ctx->reg_ptrs[inst->dest_idx];

    const i32 st_batch_a = SF_GET_STRIDE_S1(inst);
    const i32 st_batch_d = SF_GET_STRIDE_D(inst);

    int rows = a_shape[a_ndim - 2];
    int cols = a_shape[a_ndim - 1];
    
    if (rows != cols) return; // Only square matrices

    for (size_t b = 0; b < ctx->batch_size; ++b) {
        u8* da = base_a + b * st_batch_a;
        u8* dd = base_d + b * st_batch_d;

        int32_t s1 = 4;
        int32_t s0 = cols * s1;
        int32_t d1 = 4;
        int32_t d0 = rows * d1;

        if (rows == 3) {
            sf_mat3 m;
            for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
                m.m[r * 3 + c] = *(f32*)(da + r * s0 + c * s1);
            }
            sf_mat3 res = sf_mat3_inverse(m);
            for(int r=0; r<3; ++r) for(int c=0; c<3; ++c) {
                *(f32*)(dd + r * d0 + c * d1) = res.m[r * 3 + c];
            }
        } else if (rows == 4) {
            sf_mat4 m;
            for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) {
                m.m[r * 4 + c] = *(f32*)(da + r * s0 + c * s1);
            }
            sf_mat4 res = sf_mat4_inverse(m);
            for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) {
                *(f32*)(dd + r * d0 + c * d1) = res.m[r * 4 + c];
            }
        }
    }
}

void op_JOIN(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    uint8_t d_ndim = ctx->reg_ndims[inst->dest_idx];
    int components = ctx->reg_shapes[inst->dest_idx][d_ndim - 1];
    
    u8* base_d = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* base_a = (u8*)ctx->reg_ptrs[inst->src1_idx];
    u8* base_b = (u8*)ctx->reg_ptrs[inst->src2_idx];
    u8* base_c = (inst->src3_idx != 0xFF) ? (u8*)ctx->reg_ptrs[inst->src3_idx] : NULL;
    u8* base_d_in = (inst->src4_idx != 0xFF) ? (u8*)ctx->reg_ptrs[inst->src4_idx] : NULL;

    i32 st_d = SF_GET_STRIDE_D(inst);
    i32 st_a = SF_GET_STRIDE_S1(inst);
    i32 st_b = SF_GET_STRIDE_S2(inst);
    i32 st_c = SF_GET_STRIDE_S3(inst);
    i32 st_d_in = SF_GET_STRIDE_S4(inst);

    int32_t sd = 4; // F32 strides assumed contiguous for components

    for (size_t i = 0; i < ctx->batch_size; ++i) {
        u8* cur_d = base_d + i * st_d;
        *(f32*)(cur_d + 0 * sd) = *(f32*)(base_a + i * st_a);
        *(f32*)(cur_d + 1 * sd) = *(f32*)(base_b + i * st_b);
        if (base_c && components >= 3) *(f32*)(cur_d + 2 * sd) = *(f32*)(base_c + i * st_c);
        if (base_d_in && components >= 4) *(f32*)(cur_d + 3 * sd) = *(f32*)(base_d_in + i * st_d_in);
    }
}