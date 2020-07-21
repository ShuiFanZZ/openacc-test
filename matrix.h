#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int read_mtx_header(FILE *f, int *M, int *N, int *nz);
void read_matrix_csc(FILE *f, int *Ln, int *Lnz, int **Lp, int **Li, double **Lv);
void read_matrix_csr(FILE* f, int* Ln, int* Lnz, int** Lp, int** Li, double** Lv);
void read_vector(FILE *f, int *xn, int *xnz, int **xi, double **x);