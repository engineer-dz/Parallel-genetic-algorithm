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
__kernel void generation(__global double* d_F, __global double* d_D, __global int* d_permutation_parents, __global double* d_X_parents, __global double* d_fitness_parents, __global int* d_permutation, __global double* d_X, __global double* d_fitness)
{
    int id_global = get_global_id(0);
    int id_group = get_group_id(0);

	/*
    for(int i = 0; i < NB_GENS; i++)
	    d_permutation[id_group*NB_GENES + i] = d_permutation_parents[id_group*NB_GENES + i];

    for(int i = 0; i < NB_GENS; i++)
	    d_X[id_group*NB_GENES*NB_GENES + i*NB_GENES + d_permutation[id_group*NB_GENES + i]] = 1;
	*/

    d_fitness[id_global] = 1.0;
}
