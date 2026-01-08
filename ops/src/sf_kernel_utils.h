#ifndef SF_KERNEL_UTILS_H
#define SF_KERNEL_UTILS_H

#include <sionflow/isa/sf_instruction.h>
#include <sionflow/base/sf_math.h>
#include "sf_ops_internal.h"
#include <math.h>
#include <string.h>
#include <sionflow/isa/sf_exec_ctx.h>

// --- Stride Inference ---

#define SF_GET_STRIDE_D(inst)  (ctx->reg_strides[(inst)->dest_idx][(ctx->ndim > 0) ? ctx->ndim - 1 : 0])
#define SF_GET_STRIDE_S1(inst) (ctx->reg_strides[(inst)->src1_idx][(ctx->ndim > 0) ? ctx->ndim - 1 : 0])
#define SF_GET_STRIDE_S2(inst) (ctx->reg_strides[(inst)->src2_idx][(ctx->ndim > 0) ? ctx->ndim - 1 : 0])
#define SF_GET_STRIDE_S3(inst) (ctx->reg_strides[(inst)->src3_idx][(ctx->ndim > 0) ? ctx->ndim - 1 : 0])
#define SF_GET_STRIDE_S4(inst) (ctx->reg_strides[(inst)->src4_idx][(ctx->ndim > 0) ? ctx->ndim - 1 : 0])

// --- Macros: Optimized Kernel Definitions ---

#define SF_SAFE_F32(x) (isfinite((float)(x)) ? (f32)(x) : 0.0f)

#endif // SF_KERNEL_UTILS_H