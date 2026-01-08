#include <sionflow/ops/sf_ops_core.h>
#include "sf_kernel_utils.h"
#include <sionflow/isa/sf_opcodes.h>
#include <sionflow/base/sf_math.h>
#include <sionflow/isa/sf_exec_ctx.h>
#include "sf_ops_internal.h"
#include <math.h>

/**
 * SionFlow Atomic Kernels
 * Note: Automatic kernels are now in sf_ops_auto.c
 */

// --- Vector Math (Custom Kernels) ---

void op_SMOOTHSTEP(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    size_t sz = ctx->batch_size;
    
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* x_ptr = (u8*)ctx->reg_ptrs[inst->src2_idx];
    u8* e_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    
    i32 st0 = SF_GET_STRIDE_D(inst);
    i32 st1 = SF_GET_STRIDE_S1(inst);
    i32 st2 = SF_GET_STRIDE_S2(inst);

    for (size_t i = 0; i < sz; ++i) {
        f32 e0 = ((f32*)e_ptr)[0];
        f32 e1 = ((f32*)e_ptr)[1];
        f32 val = *(f32*)x_ptr;
        
        f32 span = e1 - e0;
        if (fabsf(span) < 1e-6f) span = (span < 0) ? -1e-6f : 1e-6f;

        f32 t = (val - e0) / span;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        
        *(f32*)d_ptr = SF_SAFE_F32(t * t * (3.0f - 2.0f * t));
        
        x_ptr += st2;
        e_ptr += st1;
        d_ptr += st0;
    }
}

// --- Reduction ---

void op_SUM(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    size_t sz = ctx->batch_size;
    
    f32 sum = 0;
    u8* s_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    i32 st1 = SF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        sum += *(f32*)s_ptr;
        s_ptr += st1;
    }
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    *d_ptr += sum; // Accumulate partial sum
}

void op_SUM_STABLE(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    size_t sz = ctx->batch_size;
    
    f64 sum = 0; // Use double for precision
    u8* s_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    i32 st1 = SF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        sum += (f64)(*(f32*)s_ptr);
        s_ptr += st1;
    }
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    *d_ptr += (f32)sum; // Accumulate partial sum
}

void op_SIZE(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    uint8_t ndim = ctx->reg_ndims[inst->src1_idx];
    const int32_t* shape = ctx->reg_shapes[inst->src1_idx];
    size_t count = 1;
    for (int i = 0; i < ndim; ++i) {
        count *= (shape[i] > 0 ? (size_t)shape[i] : 1);
    }
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    *d_ptr = (f32)count;
}