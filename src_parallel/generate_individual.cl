/*------------------------------------------------------------------------------
 **
 ** Kernel:  generate_individual
 **
 ** Purpose: Generate random individuals
 **
 ** Input:
 **
 ** Output:
 **
 */

// Functions
uint xorshift32(uint seed);
int rand_int(uint random, int n);
void compute_fitness(__global double *d_F, __global double *d_D, __global double* d_A, __global double* d_B, __global double* d_C, __global double *d_X, __global double *d_fitness);
void update_X(__global double* d_X, __global int* d_permutation);

// Kernel
__kernel void generate_individual(__global double *d_F, __global double *d_D, __global double* d_A, __global double* d_B, __global double* d_C, __global int *d_permutation, __global double *d_X, __global double *d_fitness)
{
    // d_X initialized with 0
    // We assume d_permutation is initialized in the CPU
    int id_global = get_global_id(0);
    int id_local = get_local_id(0);
    int id_group = get_group_id(0);

    // Set the permutation matrix when d_permutation is computed
    int permut_id = d_permutation[id_global];   // Get permutation as a local ID
    // id_group*get_local_size(0)*get_local_size(0) to get at the correct group
    // id_local*get_local_size(0) + k the wanted local id
    update_X(d_X, d_permutation);
    compute_fitness(d_F, d_D, d_A, d_B, d_C, d_X, d_fitness);
}

// Our random number generator
// https://en.wikipedia.org/wiki/Xorshift
uint xorshift32(uint seed)
{
    uint x = seed;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    return x;
}

// Generate random integers < n
int rand_int(uint random, int n)
{
    return random % n;
}

// A function that updates the permutation matrix X using the permutation array
void update_X(__global double* d_X, __global int* d_permutation) {
    int size_work_group = get_local_size(0); // size of a work group (number of items in a work group); number of genes
// The different ids
    int id_global = get_global_id(0);
    int id_local = get_local_id(0);
    int id_group = get_group_id(0);

    // We fill the matrices with 0's
    for(int i = 0; i < size_work_group; i++) {
        d_X[id_group*size_work_group*size_work_group + id_local*size_work_group + i] = 0;
    }
    int permut_id = d_permutation[id_group*size_work_group + id_local];   // Get permutation as a local ID
    d_X[id_group*size_work_group*size_work_group + id_local*size_work_group + permut_id] = 1; // We fill the positions corresponding to the permutation with 1's
}

// A function that computes the fitnesses
void compute_fitness(__global double *d_F, __global double *d_D, __global double* d_A, __global double* d_B, __global double* d_C, __global double *d_X, __global double *d_fitness) {
    int size_work_group = get_local_size(0); // size of a work group (number of items in a work group); number of genes
// The different ids
    int id_global = get_global_id(0);
    int id_local = get_local_id(0);
    int id_group = get_group_id(0);

    double tmp; // Stores intermediate values for matrix multiplication
    // A = F*X
    for (int j = 0; j < size_work_group; j++) {
        tmp = 0.0f;
        for (int k = 0; k < size_work_group; k++) {
            tmp += d_F[id_local*size_work_group+k] * d_X[id_group*size_work_group*size_work_group + k*size_work_group+j];
        }
        d_A[id_group*size_work_group*size_work_group + id_local*size_work_group+j] = tmp;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // B = X*D
    for (int j = 0; j < size_work_group; j++) {
        tmp = 0.0f;
        for (int k = 0; k < size_work_group; k++) {
            tmp += d_X[id_group*size_work_group*size_work_group + id_local*size_work_group+k] * d_D[k*size_work_group+j];
        }
        d_B[id_group*size_work_group*size_work_group + id_local*size_work_group+j] = tmp;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // C = (X*D)transpose
    for(int j = 0; j < size_work_group; j++)
        d_C[id_group*size_work_group*size_work_group + id_local*size_work_group + j] = d_B[id_group*size_work_group*size_work_group + j*size_work_group + id_local];

    barrier(CLK_GLOBAL_MEM_FENCE);

    // B = A*C
    for (int j = 0; j < size_work_group; j++) {
        tmp = 0.0f;
        for (int k = 0; k < size_work_group; k++) {
            tmp += d_A[id_group*size_work_group*size_work_group + id_local*size_work_group+k] * d_C[id_group*size_work_group*size_work_group + k*size_work_group+j];
        }
        d_B[id_group*size_work_group*size_work_group + id_local*size_work_group+j] = tmp;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // fitness = trace(B)
    if(id_local == 0) {
        double trace = 0;
        for(int i = 0; i < size_work_group; i++) {
            trace += d_B[id_group*size_work_group*size_work_group + i*size_work_group + i];
        }
        d_fitness[id_group] = trace;
    }
}
