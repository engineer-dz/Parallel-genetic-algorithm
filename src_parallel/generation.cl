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
void xorshift32(uint *seed)
{
    uint x = *seed;

    *seed ^= *seed << 13;
    *seed ^=  *seed>> 17;
    *seed ^= *seed << 5;
}


int rand_int(uint random, int n)
{
    return random % n;
}



// Kernel
__kernel void generation(__global double* d_F, __global double* d_D, __global uint* d_operator_probability, __global int* d_permutation_parents, __global double* d_X_parents, __global double* d_fitness_parents, __global int* d_permutation, __global double* d_X, __global double* d_fitness)
{
    int id_global = get_global_id(0);
    int id_local = get_local_id(0);
    int id_group = get_group_id(0);

    uint seed = d_operator_probability[id_group];

    // We first copy the values of the parents into the child
    d_permutation[id_group*NB_GENES + id_local] = d_permutation_parents[id_group*NB_GENES + id_local];

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Serial code, the first item will copy half of the second parent into the child
    if(id_local == 0) {
	//Randomly select a second parent from the whole population
        int parent = rand_int(seed, get_num_groups(0));
        //int parent = 0; // To check if the crossover works, use the first individual with predifined site, the end of the permutations should all be similar
        // Doesn't work
        //xorshift32(&seed)
	// Will we copy the first half or the second? Random
        int which_first = rand_int(seed, 2);
	// Where should we cut the two halves? Random
        int site = 0;
        int k = 1; // An iterator to keep track of the position we are modifying
        for(int i = 0; i < NB_GENES; i++) {
            int j = 0;
            int value = d_permutation_parents[parent*NB_GENES + i];
            while( (d_permutation[id_group*NB_GENES + j] != value) & (j <= site) )
                j++;
            if( j > site) {
                d_permutation[id_group*NB_GENES + site + k] = value;
                k++;
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // We empty the X matrix with zeroes, a row for each item
    for(int i = 0; i < NB_GENES; i++) {
        d_X[id_group*NB_GENES*NB_GENES + id_local*NB_GENES + i] = 0;
    }

    // We put a 1 value for each row
    int permut_id = d_permutation[id_group*NB_GENES + id_local];   // Get permutation as a local ID
    d_X[id_group*NB_GENES*NB_GENES + id_local*NB_GENES + permut_id] = 1;

    d_fitness[id_group] = 4.0; // Just a test to check that we have entered the kernel; change the value from time to time
}
