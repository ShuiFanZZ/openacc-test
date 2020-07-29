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

__global__ void
spmv_csr_vector_kernel(const int num_rows,
                       const int *ptr,
                       const int *indices,
                       const double *data,
                       const double *x,
                       double *y)
{
    __shared__ volatile double vals[1024];
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    int warp_id = thread_id / 32;                          // global warp index
    int lane = thread_id & (32 - 1);                       // thread index within the warp
    // each warp process one row
    int row = warp_id;
    vals[threadIdx.x] = 0;
    
    if (row < num_rows)
    {
        int row_start = ptr[row];
        int row_end = ptr[row + 1];
        // compute running sum per thread
        for (int jj = row_start + lane; jj < row_end; jj += 32)
            vals[threadIdx.x] += data[jj] * x[indices[jj]];

        // reduction within warp to the first thread
        if (lane < 16)
            vals[threadIdx.x] += vals[threadIdx.x + 16];
        if (lane < 8)
            vals[threadIdx.x] += vals[threadIdx.x + 8];
        if (lane < 4)
            vals[threadIdx.x] += vals[threadIdx.x + 4];
        if (lane < 2)
            vals[threadIdx.x] += vals[threadIdx.x + 2];
        if (lane < 1)
            vals[threadIdx.x] += vals[threadIdx.x + 1];
        // first thread writes the result
        if (lane == 0)
            y[row] += vals[threadIdx.x];
            
    }
}

void spmv_csr_cuda(int n, int *Ap, int *Ai, double *Ax, double *x, double *y)
{
    int nz = Ap[n];
    int *csrRowPtrA, *csrColIndA;
    double *valA, *valx, *valy; 
    cudaMalloc((void **)&csrRowPtrA, (n + 1) * sizeof(int));
    cudaMalloc((void **)&csrColIndA, nz * sizeof(int));
    cudaMalloc((void **)&valA, nz * sizeof(double));
    cudaMalloc((void **)&valx, n * sizeof(double));
    cudaMalloc((void **)&valy, n * sizeof(double));
    cudaMemcpy(csrRowPtrA, Ap, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColIndA, Ai, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(valA, Ax, nz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(valx, x, n * sizeof(double), cudaMemcpyHostToDevice);

    int num_threads = n * 32; // Each warp (32 threads) takes one row
    int Blocks = (num_threads - 1) / THREAD_PER_BLOCK + 1;
    spmv_csr_vector_kernel<<<Blocks, THREAD_PER_BLOCK>>>(n, csrRowPtrA, csrColIndA, valA, valx, valy);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaMemcpy(y, valy, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(csrRowPtrA);
    cudaFree(csrColIndA);
    cudaFree(valA);
    cudaFree(valx);
    cudaFree(valy);
}