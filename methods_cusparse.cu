#include "methods_cusparse.h"

void spmv_csr_cusparse(int n, int *Ap, int *Ai, double *Ax, double *x, double *y)
{
    cusparseHandle_t handle = 0;
    cusparseCreate(&handle);
    cusparseStatus_t status;

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
    cudaMemcpy(valy, y, n * sizeof(double), cudaMemcpyHostToDevice);

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    cusparseCreateCsr(&matA, n, n, nz,
                      csrRowPtrA, csrColIndA, valA,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    
    cusparseCreateDnVec(&vecX, n, valx, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, n, valy, CUDA_R_64F);

    double alpha = 1.0;
    double beta  = 0.0;
    size_t buffersize = 0;
    status = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_CSRMV_ALG1, &buffersize);
    
    void *buffer;
    cudaMalloc(&buffer, buffersize);

    status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_CSRMV_ALG1, buffer);

    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        printf("CuSparse csrmv failed.\n");
        exit(-1);
    }
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    
    cudaMemcpy(y, valy, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(buffer);
    cudaFree(csrRowPtrA);
    cudaFree(csrColIndA);
    cudaFree(valA);
    cudaFree(valx);
    cudaFree(valy);
}