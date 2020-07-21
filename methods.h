#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include <math.h>
#include <string.h>

/*
* Lower triangular solver Lx=b
*/
void sptrsv_csr(int n, int* Lp, int* Li, double* Lx, double* x);

int build_levelSet_CSR(int n, int* Lp, int* Li,
    int** lv_Ptr, int** lv_Set);
void sptrsv_csr_levelset_openacc(int n, int nz, int* Lp, int* Li, double* Lx,
    double* x,
    int levels, int* levelPtr,
    int* levelSet);


enum visit{
    UNVISITED=0,
    VISITED=1,
    QUEUED=2
};
/*
* Sparse matrix-vector multiply: y = A*x
* Output:
* y : is a dense vector that stores the result of multiplication
*/
int spmv_csc(int n, int *Ap, int *Ai, double *Ax,
             double *x, double *y);

int spmv_csr(int n, int* Ap, int* Ai, double* Ax,
    double* x, double* y);

/*
* Dense matrix addition
*/
void matrix_add_openacc(int matrix_size, double **A, double **B, double **C);
void vector_add_openacc(int vector_size, double *A, double *B, double *C);