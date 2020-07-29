#include <stdio.h>
#include "cuda_runtime.h"
#include <cusparse.h>


#if __cplusplus
extern "C" {
#endif

void spmv_csr_cusparse(int n, int *Ap, int *Ai, double *Ax, double *x, double *y);

#if __cplusplus
}
#endif