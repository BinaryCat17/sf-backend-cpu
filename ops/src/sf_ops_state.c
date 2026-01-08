#include <sionflow/ops/sf_ops_core.h>
#include "sf_kernel_utils.h"
#include <sionflow/isa/sf_opcodes.h>
#include <sionflow/base/sf_math.h>
#include "sf_ops_internal.h"
#include <string.h>

// All data movement operations (COPY, SLICE, RESHAPE) have been removed 
// from the execution backend. They are now handled as Zero-Copy views 
// by the Engine and Compiler.
