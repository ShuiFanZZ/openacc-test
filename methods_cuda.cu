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

__global__ void
spmv_csr_scalar_kernel(const int num_rows,
                       const int *ptr,
                       const int *indices,
                       const double *data,
                       const double *x,
                       double *y)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread index
    
    if (thread_id < num_rows)
    {
        double val = 0;
        int row_start = ptr[thread_id];
        int row_end = ptr[thread_id + 1];
        for (int jj = row_start; jj < row_end; jj ++)
            val += data[jj] * x[indices[jj]];

        y[thread_id] += val; 
    }
}



void spmv_csr_cuda(int n, int *Ap, int *Ai, double *Ax, double *x, double *y)
{
    int nz = Ap[n];
    int *csrRowPtrA, *csrColIndA;
    double *valA, *valx, *valy; 
    float transfer_in, computation_time, transfer_out; // timing values
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMalloc((void **)&csrRowPtrA, (n + 1) * sizeof(int));
    cudaMalloc((void **)&csrColIndA, nz * sizeof(int));
    cudaMalloc((void **)&valA, nz * sizeof(double));
    cudaMalloc((void **)&valx, n * sizeof(double));
    cudaMalloc((void **)&valy, n * sizeof(double));
    cudaMemcpy(csrRowPtrA, Ap, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColIndA, Ai, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(valA, Ax, nz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(valx, x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_in, start, stop);


    cudaEventRecord(start);
    int num_threads = n * 32; // Each warp (32 threads) takes one row
    int Blocks = (num_threads - 1) / THREAD_PER_BLOCK + 1;
    spmv_csr_vector_kernel<<<Blocks, THREAD_PER_BLOCK>>>(n, csrRowPtrA, csrColIndA, valA, valx, valy);
    
    // int num_threads = n;
    // int Blocks = (num_threads - 1) / THREAD_PER_BLOCK + 1;
    // spmv_csr_scalar_kernel<<<Blocks, THREAD_PER_BLOCK>>>(n, csrRowPtrA, csrColIndA, valA, valx, valy);
    
    cudaError_t error = cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaEventRecord(start);
    cudaMemcpy(y, valy, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_out, start, stop);
    
    // print timing results
    printf("%15.2f %15.2f %15.2f\n", transfer_in,
            computation_time, transfer_out);

    cudaFree(csrRowPtrA);
    cudaFree(csrColIndA);
    cudaFree(valA);
    cudaFree(valx);
    cudaFree(valy);
}

int spmv_csr_adaptive_rowblocks(int *ptr, int totalRows, int *rowBlocks)
{
    rowBlocks[0] = 0;
    int sum = 0;
    int last_i = 0;
    int ctr = 1;
    for (int i = 1; i <= totalRows; i++)
    {
        // Count non-zeroes in this row
        sum += ptr[i] - ptr[i - 1];
        if (sum == BLOCK_DIM)
        {
            // This row fills up LOCAL_SIZE
            last_i = i;
            rowBlocks[ctr++] = i;
            sum = 0;
        }
        else if (sum > BLOCK_DIM)
        {
            if (i - last_i > 1)
            {
                // This extra row will not fit
                rowBlocks[ctr++] = i - 1;
                i--;
            }
            else if (i - last_i == 1)
                // This one row is too large
                rowBlocks[ctr++] = i;

            last_i = i;
            sum = 0;
        }
    }
    rowBlocks[ctr++] = totalRows;
    return ctr;
}

__global__ void spmv_csr_adaptive_kernel(double *d_val, double *d_vector, int *d_cols, int *d_ptr, int N, int *d_rowBlocks, double *d_out)
{
    int startRow = d_rowBlocks[blockIdx.x];
    int nextStartRow = d_rowBlocks[blockIdx.x + 1];
    int num_rows = nextStartRow - startRow;
    int i = threadIdx.x;
    __shared__ volatile double LDS[BLOCK_DIM];
    
    // If the block consists of more than one row then run CSR Stream
    if (num_rows > 1)
    {
        int nnz = d_ptr[nextStartRow] - d_ptr[startRow];
        int first_col = d_ptr[startRow];

        // Each thread writes to shared memory
        if (i < nnz)
        {
            LDS[i] = d_val[first_col + i] * d_vector[d_cols[first_col + i]];
        }
        __syncthreads();

        // Threads that fall within a range sum up the partial results
        for (int k = startRow + i; k < nextStartRow; k += blockDim.x)
        {
            double temp = 0;
            for (int j = d_ptr[k] - first_col; j < d_ptr[k + 1] - first_col; j++)
            {
                temp = temp + LDS[j];
            }
            d_out[k] = temp;
        }

    }
    // If the block consists of only one row then run CSR Vector
    else if(num_rows == 1)
    {
        // Thread ID in warp
        int rowStart = d_ptr[startRow];
        int rowEnd = d_ptr[nextStartRow];

        double sum = 0;

        // Use all threads in a warp to accumulate multiplied elements
        for (int j = rowStart + i; j < rowEnd; j += BLOCK_DIM)
        {
            int col = d_cols[j];
            sum += d_val[j] * d_vector[col];
        }

        LDS[i] = sum;
        __syncthreads();

        // Reduce partial sums
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            __syncthreads();
            if (i < stride)
                LDS[i] += LDS[i + stride];
        }
        // Write result
        if (i == 0)
            d_out[startRow] = LDS[i];
    }
}

void spmv_csr_cuda_opt(int n, int *Ap, int *Ai, double *Ax, double *x, double *y)
{
    int nz = Ap[n];
    int *csrRowPtrA, *csrColIndA;
    double *valA, *valx, *valy; 
    float transfer_in, computation_time, transfer_out; // timing values
    int *rowBlocks, *d_rowBlocks;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    rowBlocks = (int *)malloc(n * sizeof(int));

    cudaEventRecord(start);
    cudaMalloc((void **)&csrRowPtrA, (n + 1) * sizeof(int));
    cudaMalloc((void **)&csrColIndA, nz * sizeof(int));
    cudaMalloc((void **)&valA, nz * sizeof(double));
    cudaMalloc((void **)&valx, n * sizeof(double));
    cudaMalloc((void **)&valy, n * sizeof(double));
    cudaMemcpy(csrRowPtrA, Ap, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColIndA, Ai, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(valA, Ax, nz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(valx, x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_in, start, stop);


    cudaEventRecord(start);

    int num_rowBlocks = spmv_csr_adaptive_rowblocks(Ap, n,rowBlocks);
    cudaMalloc(&d_rowBlocks, (num_rowBlocks + 1) * sizeof(int));
    cudaMemcpy(d_rowBlocks, rowBlocks, (num_rowBlocks + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    spmv_csr_adaptive_kernel<<<num_rowBlocks, BLOCK_DIM>>>(valA, valx, csrColIndA, csrRowPtrA, n, d_rowBlocks, valy);
    
    cudaError_t error = cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaEventRecord(start);
    cudaMemcpy(y, valy, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_out, start, stop);
    
    // print timing results
    printf("%15.2f %15.2f %15.2f\n", transfer_in,
            computation_time, transfer_out);

    cudaFree(csrRowPtrA);
    cudaFree(csrColIndA);
    cudaFree(valA);
    cudaFree(valx);
    cudaFree(valy);
    cudaFree(d_rowBlocks);
    free(rowBlocks);
}
