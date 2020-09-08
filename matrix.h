#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum DATA_DIRECTION{
    row_major = 0,
    column_major = 1,
};

int read_mtx_header(FILE *f, int *M, int *N, int *nz);
void read_matrix_csc(FILE *f, int *Ln, int *Lnz, int **Lp, int **Li, double **Lv);
void read_matrix_csr_sym(FILE* f, int* Ln, int* Lnz, int** Lp, int** Li, double** Lv);
void read_matrix_csr(FILE* f, int* Ln, int* Lnz, int** Lp, int** Li, double** Lv);
void read_vector(FILE *f, int *xn, int *xnz, int **xi, double **x);
void create_matrix_csr(int N, int* Lnz, int** Lp, int** Li, double** Lv, int* Bn, int** Bp, int** Br, int ** Bc);
void csr_to_bcsr(int N, double *Ax, int *Ai, int *Ap, double** Bx, int** Bi, int** Bp, int**Br, int block_size, int data_order);
