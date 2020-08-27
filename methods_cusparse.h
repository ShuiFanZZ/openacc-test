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

#if __cplusplus
extern "C" {
#endif

void spmv_bcsr_cusparse(int n, int nb, int blockDim, int *Bp, int *Bi, double *Bx, double *x, double *y, int direction);

#if __cplusplus
}
#endif