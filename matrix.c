#include "matrix.h"
#define MAX_LEN 2048
#define MIN_LEN 32

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


void create_matrix_csr(int N, int* Lnz, int** Lp, int** Li, double** Lv, int* Bn, int** Bp, int** Br, int ** Bc)
{
    int current_row = 0;
    int nz = 0;

    *Lv = (double*)malloc(N * MAX_LEN * sizeof(double));
    *Li = (int*)malloc(N * MAX_LEN * sizeof(int));
    *Lp = (int*)malloc((N + 1) * sizeof(int));
    *Bp = (int*)malloc((N + 1) * sizeof(int));
    *Br = (int*)malloc(N * sizeof(int));
    *Bc = (int*)malloc(N * sizeof(int));
    int num_blocks = 0;
    (*Bp)[0] = 0;
    (*Lp)[0] = 0;

    while (current_row < N)
    {   
        int m,n;
        if (current_row > N - MAX_LEN)
        {
            m = N - current_row;
        }
        else
        {
            m = (rand() % (MAX_LEN - MIN_LEN + 1)) + MIN_LEN;
        }
        n = (rand() % (MAX_LEN - MIN_LEN + 1)) + MIN_LEN;
        int start_index = 0;
        if (n > N)
        {
            n = N;
        }else
        {
            start_index = rand() % (N - n);
        }
        
        
        (*Br)[num_blocks] = m;
        (*Bc)[num_blocks] = n;


        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                (*Li)[nz] = start_index + j;
                (*Lv)[nz] = (double)rand() / RAND_MAX * 2.0 - 1.0;
                nz++;
            }
            current_row ++;
            (*Lp)[current_row] = nz;
            
        }
        
        num_blocks ++;
        (*Bp)[num_blocks] = current_row;
    }
    

    *Lnz = nz;
    *Bn = num_blocks;

}


void csr_to_bcsr(int N, double *Ax, int *Ai, int *Ap, double** Bx, int** Bi, int** Bp, int**Br, int block_size, int data_order)
{
    int num_block_row = (N - 1) / block_size + 1;
    *Bp = (int*)malloc((num_block_row + 1) * sizeof(int));
    (*Bp)[0] = 0;

    int **block_map = (int**)malloc(num_block_row * sizeof(int*));

    // Count the number of blocks
    int num_block = 0;
    for (int block_row = 0; block_row < num_block_row; block_row++)
    {
        block_map[block_row] = (int*)malloc(num_block_row * sizeof(int));
        memset(block_map[block_row], 0, sizeof(int) * num_block_row);

        for (int row = block_row * block_size; row < (block_row + 1) * block_size && row < N; row++)
        {
            int start = Ap[row];
            int end = Ap[row + 1];
            for (int i = start; i < end; i++)
            {
                int col = Ai[i];
                int current_block_index = col / block_size;
                block_map[block_row][current_block_index] = 1;
            }
        }

        int row_num_block = 0;
        for (int i = 0; i < num_block_row; i++)
        {
            if (block_map[block_row][i] == 1)
            {
                row_num_block++;
                num_block++;
                block_map[block_row][i] = row_num_block;
            }
            
        }


        (*Bp)[block_row + 1] = num_block;
    }
    
    *Bx = (double*)malloc(num_block * block_size * block_size * sizeof(double));
    memset(*Bx, 0, num_block * block_size * block_size * sizeof(double));

    for (int row = 0; row < N; row++)
    {
        int start = Ap[row];
        int end = Ap[row + 1];
        int block_row = row / block_size;
        int inner_row = row % block_size;

        for (int i = start; i < end; i++)
        {
            int col = Ai[i];
            int inner_col = col % block_size;
            int block_col = col / block_size;
            int offset = ((*Bp)[block_row] + block_map[block_row][block_col] - 1) * block_size * block_size;
            
            if (data_order == row_major)
            {
                (*Bx)[offset + inner_row * block_size + inner_col] = Ax[i];
            }else
            {
                (*Bx)[offset + inner_col * block_size + inner_row] = Ax[i];
            }


        }
        
    }
    
    *Bi = (int*)malloc(num_block * sizeof(int));
    *Br = (int*)malloc(num_block * sizeof(int));
    int index = 0;
    for (int i = 0; i < num_block_row; i++)
    {
        for (int j = 0; j < num_block_row; j++)
        {
            if (block_map[i][j] != 0)
            {
                (*Bi)[index] = j;
                (*Br)[index] = i;
                index ++;
            }
            
        }
        free(block_map[i]);
    }

    free(block_map);
    if (index != num_block)
    {
        printf("Error!\n");
        exit(-1);
    }
    

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