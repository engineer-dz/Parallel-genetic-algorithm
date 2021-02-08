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


// Struct definition
/*
typedef struct Individual
{
    int N = NB_GENES;
    int permutation[NB_GENES];    // Permutation vector
    double X[NB_GENES * NB_GENES];   // Permutation matrix
    double fitness;
} Individual;


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


void shuffle(Individual *I, uint seed)
{
    int n = I->N;

    for (int i = n-1; i>= 0; i--) {
        // Generate a random position
        // Check, i+1 or n+1?
        seed = xorshift32(seed);
        int j = rand_int(seed, i+1);

        int tmp = I->permutation[i];
        I->permutation[i] = I->permutation[j];
        I->permutation[j] = tmp;
    }
}


void construct_matrix(Individual *I)
{
    int n = I->N;

    for (int i = 0; i < I->N; i++) {
        for (int j = 0; j < I->N; j++) {
            I->X[i*n + j] = 0;
        }
    }

    // Put the components to 1
    int k;
    for (int i = 0; i < I->n; i++) {
        k = I->permutation[i];
        I->X[i*n + k] = 1;
    }
}


void mat_mul(int N, double *A, double *B, double **C)
{
	double tmp;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			tmp = 0.0f;
			for (int k = 0; k < N; k++) {
				tmp += A[i*N+k] * B[k*N+j];
			}
			*C[i*N+j] = tmp;
		}
	}
}


void mat_transpose(int N, double *A, double **At)
{
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			*At[i*N + j] = A[j*N + i];
}


double mat_trace(int N, double *A)
{
	double tmp = 0;

	for(int i = 0; i < N; i++)
		tmp += A[i*N + i];

	return tmp;
}
*/



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
    barrier(CLK_GLOBAL_MEM_FENCE);  // Check that it does what we want

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

    d_fitness[id_global] = trace;
}