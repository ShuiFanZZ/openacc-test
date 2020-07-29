#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "getopt.h"
#include "methods.h"
#include "methods_cuda.h"
#include "methods_cusparse.h"
#define DIFF 0.00001
#define Times 5

enum MODE
{
    NAIVE = 0,
    OPENACC = 1,
    CUDA = 2,
};

enum Algorithm
{
    SPARSE_SOLVE=0,
    ADDITION=1,
    ADDITION_VECTOR=2,
    SPARSE_MV=3,
};

int compare (const void* a, const void* b){
    if (*(double*)a > *(double*)b)
    {
        return 1;
    }
    return -1;
    
}

double find_median(double* array, int n){
    qsort(array, n, sizeof(double), compare);
    return array[Times/2];
}

int double_match(double a, double b){
    if (a -b < DIFF && a - b > - DIFF)
    {
        return 0;
    }
    double d = (a - b) / a;
    if (d > DIFF || d < - DIFF)
    {
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{

    int mode = NAIVE;
    int algorithm = SPARSE_SOLVE;
    int write_output = 0;
    int matrix_size = 0;

    char *matrix_file = NULL, *vector_file = NULL;
    FILE *f_matrix, *f_vector;
    char *output = NULL;

    int option;
    while ((option = getopt(argc, argv, "MOCSV:A:L:b:o:")) != -1)
    {
        switch (option)
        {
        case 'L':
            matrix_file = optarg;
            break;
        case 'b':
            vector_file = optarg;
            break;
        case 'A':
            algorithm = ADDITION;
            matrix_size = atoi(optarg);
            break;
        case 'V':
            algorithm = ADDITION_VECTOR;
            matrix_size = atoi(optarg);
            break;
        case 'S':
            algorithm = SPARSE_SOLVE;
            break;
        case 'M':
            algorithm = SPARSE_MV;
            break;
        case 'C':
            mode = CUDA;
            break;
        case 'O':
            mode = OPENACC;
            break;
        case 'o':
            write_output = 1;
            output = optarg;
            break;
        default:
            printf("Correct Usage: ./solve (-B/-D/-P [num_threads]) -L [matrix.mtx] -b [vector.mtx]\n");
            exit(1);
        }
    }

    int Ln = 0, Lnz = 0, xn = 0, xnz = 0;
    int *Lp, *Li, *xi;
    double *Lv, *x, *b;

    double **A, **B, **C;
    double *u, *v, *w;

    if (algorithm == SPARSE_SOLVE || algorithm == SPARSE_MV)
    {
        f_matrix = fopen(matrix_file, "r");

        if (f_matrix == NULL)
        {
            printf("Cannot read input file!\n");
            exit(2);
        }

        read_matrix_csr(f_matrix, &Ln, &Lnz, &Lp, &Li, &Lv);

        fclose(f_matrix);

        if (vector_file != NULL)
        {
            f_vector = fopen(vector_file, "r");
            read_vector(f_vector, &xn, &xnz, &xi, &x);
            fclose(f_vector);
        }
        else
        {
            xn = Ln;
            x = (double *)malloc(xn * sizeof(double));
            for (int i = 0; i < xn; i++)
            {
                x[i] = 1.0;
            }
        }

        if (Ln != xn)
        {
            printf("Matrix and vector do not match.\n");
            exit(3);
        }

        // copy the vector for verification
        b = (double *)malloc(xn * sizeof(double));
        memcpy(b, x, xn * sizeof(double));
    } 
    else if (algorithm == ADDITION)
    {
        A = (double**)malloc(matrix_size * sizeof(double*));
        B = (double**)malloc(matrix_size * sizeof(double*));
        C = (double**)malloc(matrix_size * sizeof(double*));
        for (int i = 0; i < matrix_size; i++)
        {
            A[i] = (double*)malloc(matrix_size * sizeof(double));
            B[i] = (double*)malloc(matrix_size * sizeof(double));
            C[i] = (double*)malloc(matrix_size * sizeof(double));
            for (int j = 0; j < matrix_size; j++)
            {
                A[i][j] = (double)rand()/RAND_MAX*2.0-1.0;
                B[i][j] = (double)rand()/RAND_MAX*2.0-1.0;
            }
            
        }
        
    }
    else if (algorithm == ADDITION_VECTOR)
    {
        u = (double *)malloc(matrix_size * matrix_size * sizeof(double));
        v = (double *)malloc(matrix_size * matrix_size * sizeof(double));
        for (int j = 0; j < matrix_size * matrix_size; j++)
        {
            u[j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
            v[j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }
        w = (double *)malloc(matrix_size * matrix_size * sizeof(double));
    }

    // Solve
    double time_msec[Times];
    double start, end;
    int *levelPtr, *levelSet;
    int levels;
    
    for (int i = 0; i < Times; i++)
    {
        //reset x
        if (algorithm == SPARSE_SOLVE)
        {
            memcpy(x, b, xn * sizeof(double));
        }
        if (algorithm == SPARSE_MV)
        {
            memset(x, 0.0, xn * sizeof(double));
        }
        
        
        if (mode == NAIVE && algorithm == SPARSE_SOLVE)
        {
            start = omp_get_wtime();
            sptrsv_csr(Ln, Lp, Li, Lv, x);
            end = omp_get_wtime();
            time_msec[i] = (end - start) * 1000;
        }
        else if (mode == OPENACC && algorithm == SPARSE_SOLVE)
        {
            start = omp_get_wtime();

            levels = build_levelSet_CSR(Ln, Lp, Li, &levelPtr, &levelSet);
            sptrsv_csr_levelset_openacc(Ln, Lnz, Lp, Li, Lv, x, levels, levelPtr, levelSet);
            
            end = omp_get_wtime();
            time_msec[i] = (end - start) * 1000;
        }
        else if (mode == CUDA && algorithm == SPARSE_SOLVE)
        {
            start = omp_get_wtime();

            levels = build_levelSet_CSR(Ln, Lp, Li, &levelPtr, &levelSet);
            sptrsv_csr_levelset_cuda(Ln, Lnz, Lp, Li, Lv, x, levels, levelPtr, levelSet);
            
            end = omp_get_wtime();
            time_msec[i] = (end - start) * 1000;
        }
        else if (mode == OPENACC && algorithm == ADDITION)
        {
            start = omp_get_wtime();
            matrix_add_openacc(matrix_size, A, B, C);
            
            end = omp_get_wtime();
            time_msec[i] = (end - start) * 1000;
        }
        else if (mode == CUDA && algorithm == ADDITION)
        {
            start = omp_get_wtime();
            matrix_add_cuda(matrix_size, A, B, C);
            
            end = omp_get_wtime();
            time_msec[i] = (end - start) * 1000;
        }
        else if (mode == OPENACC && algorithm == ADDITION_VECTOR)
        {
            start = omp_get_wtime();
            vector_add_openacc(matrix_size * matrix_size, u, v, w);
            
            end = omp_get_wtime();
            time_msec[i] = (end - start) * 1000;
        }
        else if (mode == CUDA && algorithm == ADDITION_VECTOR)
        {
            start = omp_get_wtime();
            vector_add_cuda(matrix_size * matrix_size, u, v, w);
            
            end = omp_get_wtime();
            time_msec[i] = (end - start) * 1000;
        }
        else if (mode == OPENACC && algorithm == SPARSE_MV)
        {
            start = omp_get_wtime();
            spmv_csr_openacc(Ln, Lp, Li, Lv, b, x);
            end = omp_get_wtime();
            time_msec[i] = (end - start) * 1000;
        }
        else if (mode == NAIVE && algorithm == SPARSE_MV)
        {
            start = omp_get_wtime();
            spmv_csr_cusparse(Ln, Lp, Li, Lv, b, x);
            
            end = omp_get_wtime();
            time_msec[i] = (end - start) * 1000;
        }
        else if (mode == CUDA && algorithm == SPARSE_MV)
        {
            start = omp_get_wtime();
            spmv_csr_cuda(Ln, Lp, Li, Lv, b, x);
            
            end = omp_get_wtime();
            time_msec[i] = (end - start) * 1000;
        }
    }
    
    printf("%f\n", find_median(time_msec, Times));

    //verification
    if (algorithm == SPARSE_SOLVE)
    {
        double *y = (double *)malloc(xn * sizeof(double));
        memset(y, 0.0, xn * sizeof(double));
        spmv_csr(xn, Lp, Li, Lv, x, y);
        for (int i = 0; i < xn; i++)
        {
            if (double_match(b[i], y[i]) == -1)
            {
                printf("Verification failed at index %d: expect %.19lf but get %.19lf \n",
                       i, b[i], y[i]);
                return -1;
            }
        }
    }
    else if (algorithm == SPARSE_MV)
    {
        double *y = (double *)malloc(xn * sizeof(double));
        memset(y, 0.0, xn * sizeof(double));
        spmv_csr(xn, Lp, Li, Lv, b, y);
        for (int i = 0; i < xn; i++)
        {
            if (double_match(x[i], y[i]) == -1)
            {
                printf("Verification for spmv failed at index %d: expect %.19lf but get %.19lf \n",
                       i, y[i], x[i]);
                return -1;
            }
        }
    }
    

    // Write result x to output
    if (write_output == 1 && output != NULL)
    {
        FILE *f_output;
        f_output = fopen(output, "w");

        fprintf(f_output, "%d 1\n", xn);
        for (int i = 0; i < xn; i++)
        {
            fprintf(f_output, "%.15f\n", x[i]);
        }
        fprintf(f_output, "\n");
        fclose(f_output);
    }

    return 0;
}