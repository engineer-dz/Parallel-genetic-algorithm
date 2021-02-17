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



// Struct definition
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



// Kernel
__kernel void generate_individual(__global double *F, __global double *D, __global Individual *res, const int count)
{
    Individual I;
    int n = I.N;
    int i = get_global_id(0);
    uint seed = (uint) i % 32;
    double A[n*n];  // A = F*X
    double B[n*n];  // B = X*D
    double C[n*n];  // C = Dt*Xt

    I.fitness = 0;
    
    // Create permutation matrix
    for (int i = 0; i < n; i++)
        I.permutation[i] = i;

    shuffle(&I, seed);

    // Construct associated permutation matrix
    construct_matrix(&I);

    // Evaluate trace
    mat_mul(n, F, I.X, A);  // A = F*X
    mat_mul(n, I.X, D, B);  // B = X*D
    mat_transpose();    //C = (X*D)t
    mat_mul(n, A, C, B);    // B = A*C
    I.fitness = mat_trace(n, B);

    // Write result in global memory
    if (i < count)
        res[i] = I;
}