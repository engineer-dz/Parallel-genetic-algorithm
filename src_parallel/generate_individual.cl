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

#define NB_GENES 26



// Functions
uint xorshift32(uint seed)
{
    uint x = seed;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    return x;
}


int rand_int(uint random, int n)
{
    return random % n;
}

// Kernel
__kernel void generate_individual(__global double *d_F, __global double *d_D, __global int *d_permutation, __global double *d_X, __global double *d_fitness)
{
    // d_X initialized with 0
    // We assume d_permutation is initialized in the CPU
    int id_global = get_global_id(0);
    int id_local = get_local_id(0);
    int id_group = get_group_id(0);

    // NOPE
    /*
    d_permutation[id_global] = id_local;
    barrier(CLK_GLOBAL_MEM_FENCE);
    uint seed = xorshift32(id_global);
    int id_j = rand_int(seed, id_local + 1);
    int tmp = d_permutation[id_global];
    d_permutation[id_global] = d_permutation[id_group*NB_GENES + id_j]
    d_permutation[id_group*NB_GENES + id_j] = tmp;
    */

    // Set the permutation matrix when d_permutation is computed
    int permut_id = d_permutation[id_global];   // Get permutation as a local ID
    // id_group*NB_GENES*NB_GENES to get at the correct group
    // id_local*NB_GENES + k the wanted local id
    d_X[id_group*NB_GENES*NB_GENES + id_local*NB_GENES + permut_id] = 1;     // Set the correct element to 1


    // Evaluate the trace
    double A[NB_GENES * NB_GENES];
    double B[NB_GENES * NB_GENES];
    double C[NB_GENES * NB_GENES];
    double tmp;

    // Synchronize the threads to access d_X
    barrier(CLK_GLOBAL_MEM_FENCE);

// Rewrite what's next, maybe parallelize it? at least put an if(local_id == 0)

    // A = F*X
    for (int i = 0; i < NB_GENES; i++) {
        for (int j = 0; j < NB_GENES; j++) {
            tmp = 0.0f;
            for (int k = 0; k < NB_GENES; k++) {
                tmp += d_F[i*NB_GENES+k] * d_X[id_group*NB_GENES*NB_GENES + k*NB_GENES+j];
            }
            A[i*NB_GENES+j] = tmp;
        }
    }

    // B = X*D
    for (int i = 0; i < NB_GENES; i++) {
        for (int j = 0; j < NB_GENES; j++) {
            tmp = 0.0f;
            for (int k = 0; k < NB_GENES; k++) {
                tmp += d_X[id_group*NB_GENES*NB_GENES + i*NB_GENES+k] * d_D[k*NB_GENES+j];
            }
            B[i*NB_GENES+j] = tmp;
        }
    }

    // C = (X*D)transpose
    for(int i = 0; i < NB_GENES; i++)
        for(int j = 0; j < NB_GENES; j++)
            C[i*NB_GENES + j] = B[j*NB_GENES + i];

    // B = A*C
    for (int i = 0; i < NB_GENES; i++) {
        for (int j = 0; j < NB_GENES; j++) {
            tmp = 0.0f;
            for (int k = 0; k < NB_GENES; k++) {
                tmp += A[i*NB_GENES+k] * C[k*NB_GENES+j];
            }
            B[i*NB_GENES+j] = tmp;
        }
    }

    // fitness = trace(B)
    double trace = 0;
    for(int i = 0; i < NB_GENES; i++) {
        trace += B[i*NB_GENES + i];
    }

    d_fitness[id_group] = trace;
}
