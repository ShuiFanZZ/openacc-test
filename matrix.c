#include "matrix.h"

int read_mtx_header(FILE *f, int *M, int *N, int *nz)
{
    char line[1025];
    *M = *N = *nz = 0;

    do
    {
        if (fgets(line, 1025, f) == NULL)
            return 1;
    } while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;

    else if (sscanf(line, "%d %d", M, N) == 2)
    {
        *nz = *M;
        return 1;
    }

    return -1;
}

void read_matrix_csc(FILE *f, int *Ln, int *Lnz, int **Lp, int **Li, double **Lv)
{
    int M, N, nz;
    read_mtx_header(f, &M, &N, &nz);

    *Ln = N;
    *Lnz = nz;
    *Lv = (double *)malloc(nz * sizeof(double));
    *Li = (int *)malloc(nz * sizeof(int));
    *Lp = (int *)malloc((N + 1) * sizeof(int));

    int row, col;
    double val;
    int index = 0, p = 0;
    for (int i = 0; i < nz; i++)
    {
        if(fscanf(f, "%d %d %lg\n", &row, &col, &val) != 3){
            printf("Matrix format is broken.\n");
            exit(-1);
        }
        (*Li)[i] = row - 1;
        (*Lv)[i] = val;
        if (col > index)
        {
            (*Lp)[index] = p;
            index++;
        }
        p++;
    }
    (*Lp)[N] = nz;
}

struct Entry
{
    int row;
    int col;
    double val;
    struct Entry* next;
};

void read_matrix_csr(FILE* f, int* Ln, int* Lnz, int** Lp, int** Li, double** Lv)
{
    int M, N, nz;
    read_mtx_header(f, &M, &N, &nz);

    *Ln = N;
    *Lnz = nz;
    *Lv = (double*)malloc(nz * sizeof(double));
    *Li = (int*)malloc(nz * sizeof(int));
    *Lp = (int*)malloc((N + 1) * sizeof(int));

    struct Entry **rows = (struct Entry**)malloc(N * sizeof(struct Entry*));
    for (int i = 0; i < N; i++) {
        rows[i] = NULL;
    }

    int row, col;
    double val;
    int index = 0, p = 0;
    for (int i = 0; i < nz; i++)
    {
        if (fscanf(f, "%d %d %lg\n", &row, &col, &val) != 3)
        {
            printf("Matrix format is broken.\n");
            exit(-1);
        }
        struct Entry entry;
        entry.row = row - 1;
        entry.col = col - 1;
        entry.val = val;
        entry.next = NULL;


        if (rows[entry.row] == NULL)
        {
            rows[entry.row] = malloc(sizeof(struct Entry));
            rows[entry.row]->row = entry.row;
            rows[entry.row]->col = entry.col;
            rows[entry.row]->val = entry.val;
            rows[entry.row]->next = entry.next;
        }
        else
        {
            struct Entry* entry_ptr = rows[entry.row];
            while (entry_ptr->next != NULL) {
                entry_ptr = entry_ptr->next;
            }
            entry_ptr->next = malloc(sizeof(struct Entry));
            entry_ptr->next->row = entry.row;
            entry_ptr->next->col = entry.col;
            entry_ptr->next->val = entry.val;
            entry_ptr->next->next = entry.next;
        }
    }

    int count = 0;
    for (int i = 0; i < M; i++)
    {
        (*Lp)[i] = count;
        struct Entry *entry_ptr = rows[i];

        while(entry_ptr != NULL)
        {
            (*Lv)[count] = entry_ptr->val;
            (*Li)[count] = entry_ptr->col;
            count++;
            entry_ptr = entry_ptr->next;
        }
    }

    (*Lp)[N] = nz;

}

void read_vector(FILE *f, int *xn, int *xnz, int **xi, double **x)
{
    int M, N, nz;
    int isDense = read_mtx_header(f, &M, &N, &nz);

    if (N != 1 || isDense == -1)
    {
        printf("Invalid vector.\n");
        exit(1);
    }

    *xn = M;
    *xnz = nz;
    *x = (double *)malloc(M * sizeof(double));
    memset(*x, 0.0, sizeof(double) * M);
    *xi = (int *)malloc(M * sizeof(int));

    int row, col;
    double val;
    if (isDense == 0)
    {
        for (int i = 0; i < nz; i++)
        {
            if(fscanf(f, "%d %d %lg\n", &row, &col, &val) != 3){
                printf("Sparse vector format is broken.\n");
                exit(-1);
            }

            (*x)[row - 1] = val;
            (*xi)[i] = row - 1;
        }
    }
    else
    {
        for (int i = 0; i < nz; i++)
        {
            if(fscanf(f, "%lg\n",&val) != 1){
                printf("Dense vector format is broken.\n");
                exit(-1);
            }

            (*x)[i] = val;
            (*xi)[i] = i;
        }
    }
    
}