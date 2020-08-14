#include "methods.h"



void sptrsv_csr(int n, int *Lp, int *Li, double *Lx, double *x)
{
  int i, j;
  for (i = 0; i < n; i++)
  {
    for (j = Lp[i]; j < Lp[i + 1] - 1; j++)
    {
      x[i] -= Lx[j] * x[Li[j]];
    }
    x[i] /= Lx[Lp[i + 1] - 1];
  }
}
void print_int(int *A, int l)
{
  for (int i = 0; i < l; i++)
  {
    printf("%d ", A[i]);
  }
  printf("\n");
}
void print_double(double *A, int l)
{
  for (int i = 0; i < l; i++)
  {
    printf("%f ", A[i]);
  }
  printf("\n");
}

void sptrsv_csr_levelset_openacc(int n, int nz, int *Lp, int *Li, double *Lx,
                                 double *x,
                                 int levels, int *levelPtr,
                                 int *levelSet)
{
#pragma acc data copy(x[:n]), copyin(levelPtr[:levels + 1], levelSet[:n], Lx[:nz], Li[:nz], Lp[:n + 1])
  {
    for (int l = 0; l < levels; l++)
    {
#pragma acc parallel loop
      for (int k = levelPtr[l]; k < levelPtr[l + 1]; k++)
      {
        int i = levelSet[k];
#pragma acc loop seq
        for (int j = Lp[i]; j < Lp[i + 1] - 1; j++)
        {
          x[i] -= Lx[j] * x[Li[j]];
        }
        x[i] /= Lx[Lp[i + 1] - 1];
      }
    }
  }

}

void sptrsv_csr_levelset_openacc_opt(int n, int nz, int *Lp, int *Li, double *Lx,
                                 double *x,
                                 int levels, int *levelPtr,
                                 int *levelSet)
{
#pragma acc data copy(x[:n]), copyin(levelPtr[:levels + 1], levelSet[:n], Lx[:nz], Li[:nz], Lp[:n + 1])
  {
    for (int l = 0; l < levels; l++)
    {
      int level_start = levelPtr[l];
      int level_end = levelPtr[l + 1];
#pragma acc parallel loop gang worker vector_length(32) num_workers(32) present(x, levelSet, Lx, Li, Lp)
      for (int k = level_start; k < level_end; k++)
      {
        int i = levelSet[k];
        double temp = 0;
        int row_start = Lp[i];
        int row_end = Lp[i + 1] - 1;

#pragma acc loop vector reduction(+:temp)
        for (int j = row_start; j < row_end; j++)
        {
          temp += Lx[j] * x[Li[j]];
        }
        x[i] = (x[i] - temp) / Lx[Lp[i + 1] - 1];
      }
    }
  }

}
/*
 * Simple Sparse matrix-vector multiply
 * 
 */
int spmv_csc(int n, int *Ap, int *Ai, double *Ax,
             double *x, double *y)
{
  int p, j;
  if (!Ap || !x || !y)
    return (0);
  for (j = 0; j < n; j++)
  {
    for (p = Ap[j]; p < Ap[j + 1]; p++)
    {
      y[Ai[p]] += Ax[p] * x[j];
    }
  }
  return (1);
}

int spmv_csr(int n, int *Ap, int *Ai, double *Ax,
             double *x, double *y)
{
  int p, j;
  if (!Ap || !x || !y)
    return (0);
  for (j = 0; j < n; j++)
  {
    for (p = Ap[j]; p < Ap[j + 1]; p++)
    {
      y[j] += Ax[p] * x[Ai[p]];
    }
  }
  return (1);
}

int build_levelSet_CSR(int n, int *Lp, int *Li,
                       int **lv_Ptr, int **lv_Set)
{
  int *levelSet = (int *)malloc(n * sizeof(int));
  int *inDegree = (int *)malloc(n * sizeof(int));

  int maxLevel = 0;
  int colMaxLevel = 0;
  int col;

  int *level_len = (int *)malloc(n * sizeof(int));
  memset(level_len, 0, n * sizeof(int));

  for (int i = 0; i < n; ++i)
  {
    if (Lp[i + 1] - Lp[i] == 1)
    {
      inDegree[i] = 0;
      level_len[0]++;
    }
    else
    {
      // Get the MAX degree of this row before diagonal
      // Degree of this row is MAX + 1
      col = Li[Lp[i]];
      colMaxLevel = inDegree[col];
      for (int p = Lp[i] + 1; p < Lp[i + 1] - 1; p++)
      {
        int col = Li[p];
        if (colMaxLevel < inDegree[col])
          colMaxLevel = inDegree[col];
      }
      inDegree[i] = colMaxLevel + 1;

      // Update length of this level
      level_len[inDegree[i]]++;
      if (maxLevel < inDegree[i])
        maxLevel = inDegree[i];
    }
  }

  int num_levels = maxLevel + 1;

  int *levelPtr = (int *)malloc((num_levels + 1) * sizeof(int));
  int *levelPtr_cpy = (int *)malloc((num_levels + 1) * sizeof(int));

  // Set the pointers by counting length of each level
  int count = 0;
  levelPtr[0] = count;
  levelPtr_cpy[0] = count;
  for (int i = 0; i < num_levels; i++)
  {
    count += level_len[i];
    levelPtr[i + 1] = count;
    levelPtr_cpy[i + 1] = count;
  }

  for (int i = 0; i < n; i++)
  {
    levelSet[levelPtr_cpy[inDegree[i]]] = i;
    levelPtr_cpy[inDegree[i]]++;
  }

  free(inDegree);
  free(levelPtr_cpy);
  *lv_Ptr = levelPtr;
  *lv_Set = levelSet;
  return num_levels;
}

void matrix_add_openacc(int matrix_size, double **A, double **B, double **C)
{
#pragma acc data copyin(A[:matrix_size][:matrix_size], B[:matrix_size][:matrix_size]), copy(C[:matrix_size][:matrix_size])
  {
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < matrix_size; i++)
    {
      for (int j = 0; j < matrix_size; j++)
      {
        C[i][j] = B[i][j] + A[i][j];
      }
    }
  }
}

void vector_add_openacc(int vector_size, double *A, double *B, double *C)
{
  #pragma acc data copyin(A[:vector_size], B[:vector_size]), copy(C[:vector_size])
  {
    #pragma acc parallel loop
    for (int i = 0; i < vector_size; i++)
    {
      C[i] = A[i] + B[i]; 
    }
    
  }
 
}

void spmv_csr_openacc(int n, int *Ap, int *Ai, double *Ax, double *x, double *y)
{
  int nz = Ap[n];
#pragma acc data copyin(Ap[:n+1], Ai[0:nz], Ax[0:nz], x[:n]), copy(y[:n])
{
  #pragma acc parallel loop
  for (int j = 0; j < n; j++)
  {
    for (int p = Ap[j]; p < Ap[j + 1]; p++)
    {
      y[j] += Ax[p] * x[Ai[p]];
    }
  }
}
  
}

void spmv_csr_openacc_vector(int n, int *Ap, int *Ai, double *Ax, double *x, double *y)
{
  int nz = Ap[n];
#pragma acc data copyin(Ap[:n + 1], Ai [0:nz], Ax [0:nz], x[:n]), copy(y[:n])
  {
#pragma acc parallel loop vector_length(32) num_workers(32) gang worker
    for (int j = 0; j < n; j++)
    {
      double sum = 0;
      int row_start = Ap[j];
      int row_end = Ap[j + 1];
#pragma acc loop reduction(+:sum)  vector
      for (int p = row_start; p < row_end; p++)
      {
        int index = Ai[p];
        sum += Ax[p] * x[index];
      }

      y[j] = sum;

      
    }
  }
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
        if (sum == 1024)
        {
            // This row fills up LOCAL_SIZE
            last_i = i;
            rowBlocks[ctr++] = i;
            sum = 0;
        }
        else if (sum > 1024)
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
void spmv_csr_openacc_opt(int n, int *Ap, int *Ai, double *Ax, double *x, double *y)
{
  int nz = Ap[n];
  int *rowBlocks = (int *)malloc(n * sizeof(int));
  int num_rowBlocks = spmv_csr_adaptive_rowblocks(Ap, n,rowBlocks);

  #pragma acc data copyin(Ap[:n + 1], Ai [0:nz], Ax [0:nz], x[:n], rowBlocks[:num_rowBlocks]), copy(y[:n])
  {
    #pragma acc parallel loop vector_length(1024) gang 
    for (int j = 0; j < num_rowBlocks; j++)
    {
      double vals[1024];
      int startRow = rowBlocks[j];
      int endRow = rowBlocks[j + 1];
      int num_rows = endRow - startRow;

      if (num_rows > 1)
      {
        int startIndex = Ap[startRow];
        int nnz = Ap[endRow] - startIndex;

        #pragma acc loop vector
        for (int i = 0; i < nnz; i++)
        {
          vals[i] = Ax[startIndex + i] * x[Ai[startIndex + i]];
        }

        #pragma acc loop vector
        for (int row = startRow; row < endRow; row++)
        {
          double temp = 0;
          #pragma acc loop seq
          for (int p = Ap[row] - startIndex; p < Ap[row + 1] - startIndex; p++)
          {
            temp += vals[p];
          }
          y[row] = temp;
        }
      }
      else
      {
        double sum = 0;
        int row_start = Ap[startRow];
        int row_end = Ap[endRow];
#pragma acc loop reduction(+:sum) vector
        for (int p = row_start; p < row_end; p++)
        {
          int index = Ai[p];
          sum += Ax[p] * x[index];
        }

        y[startRow] = sum;
      }
    }
  }

}


void spmv_csr_openacc_null(int n, int *Ap, int *Ai, double *Ax, double *x, double *y)
{
  int nz = Ap[n];
#pragma acc data copyin(Ap[:n+1], Ai[0:nz], Ax[0:nz], x[:n]), copy(y[:n])
{
  
}
  
}