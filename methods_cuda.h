#include <cuda_runtime.h>
#include <cuda.h>
#define THREAD_PER_BLOCK 512


#if __cplusplus
extern "C" {
#endif

void sptrsv_csr_levelset_cuda(int n, int Lnz, int* Lp, int* Li, double* Lx,
    double* x,
    int levels, int* levelPtr,
    int* levelSet);

#if __cplusplus
}
#endif

#if __cplusplus
extern "C" {
#endif

void matrix_add_cuda(int matrix_size, double **A, double **B, double **C);

#if __cplusplus
}
#endif

#if __cplusplus
extern "C" {
#endif

void vector_add_cuda(int vector_size, double *A, double *B, double *C);

#if __cplusplus
}
#endif