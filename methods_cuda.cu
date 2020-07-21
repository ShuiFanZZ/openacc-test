#include "methods_cuda.h"

__global__ void solve_level_csr(int start, int end, int *levelSet, int *Lp, int *Li, double *Lx, double *x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + start < end)
    {
        
        int i = levelSet[idx + start];
        for (int j = Lp[i]; j < Lp[i + 1] - 1; j++)
        {
            x[i] -= Lx[j] * x[Li[j]];
        }
        x[i] /= Lx[Lp[i + 1] - 1];
        
    }

}

extern "C" void sptrsv_csr_levelset_cuda(int n, int Lnz, int* Lp, int* Li, double* Lx,
    double* x,
    int levels, int* levelPtr,
    int* levelSet)
{
   
    
    int *d_Lp, *d_Li, *d_levelSet;
    double *d_Lx, *d_x;
    cudaMalloc((void **)&d_Lp, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_Li, Lnz * sizeof(int));
    cudaMalloc((void **)&d_levelSet, n * sizeof(int));
    cudaMalloc((void **)&d_Lx, Lnz * sizeof(double));
    cudaMalloc((void **)&d_x, n * sizeof(double));
    cudaMemcpy(d_Lp, Lp, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Li, Li, Lnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_levelSet, levelSet, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Lx, Lx, Lnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
    
    for (int l = 0; l < levels; l++)
    {
        int start = levelPtr[l];
        int end = levelPtr[l + 1];
        int Blocks = (end - start - 1) / THREAD_PER_BLOCK + 1;
        solve_level_csr<<<Blocks,THREAD_PER_BLOCK>>>(start, end, d_levelSet, d_Lp, d_Li, d_Lx, d_x);
        
    }
    
    cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_Lp);
    cudaFree(d_Li);
    cudaFree(d_levelSet);
    cudaFree(d_Lx);
    cudaFree(d_x);
    

}

__global__ void  add_row(int matrix_size, double* A_row, double* B_row, double* C_row)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < matrix_size)
    {
        C_row[idx] = A_row[idx] + B_row[idx];
    }
    
}

void matrix_add_cuda(int matrix_size, double **A, double **B, double **C)
{
    double *A_row, *B_row, *C_row;
    cudaMalloc((void **)&A_row, matrix_size * sizeof(double));
    cudaMalloc((void **)&B_row, matrix_size * sizeof(double));
    cudaMalloc((void **)&C_row, matrix_size * sizeof(double));
    for (int i = 0; i < matrix_size; i++)
    {
        cudaMemcpy(A_row, A[i], matrix_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(B_row, B[i], matrix_size * sizeof(double), cudaMemcpyHostToDevice);

        int Blocks = (matrix_size - 1) / THREAD_PER_BLOCK + 1;
        add_row<<<Blocks,THREAD_PER_BLOCK>>>(matrix_size, A_row, B_row, C_row);

        cudaMemcpy(C[i], C_row, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
    }
   cudaFree(A_row);
   cudaFree(B_row);
   cudaFree(C_row);

}

void vector_add_cuda(int vector_size, double *A, double *B, double *C)
{
    double *A_row, *B_row, *C_row;
    cudaMalloc((void **)&A_row, vector_size * sizeof(double));
    cudaMalloc((void **)&B_row, vector_size * sizeof(double));
    cudaMalloc((void **)&C_row, vector_size * sizeof(double));

    cudaMemcpy(A_row, A, vector_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_row, B, vector_size * sizeof(double), cudaMemcpyHostToDevice);

    int Blocks = (vector_size - 1) / THREAD_PER_BLOCK + 1;
    add_row<<<Blocks, THREAD_PER_BLOCK>>>(vector_size, A_row, B_row, C_row);

    cudaMemcpy(C, C_row, vector_size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(A_row);
    cudaFree(B_row);
    cudaFree(C_row);
}