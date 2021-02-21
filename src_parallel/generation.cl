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
unsigned int xorshift32(unsigned int seed)
{
    unsigned int x = seed;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    return x;
}


int rand_int(unsigned int random, int n)
{
    return random % n;
}



// Kernel
__kernel void generation(__global double* d_F, __global double* d_D, __global unsigned int* d_operator_probability, __global int* d_permutation_parents, __global double* d_X_parents, __global double* d_fitness_parents, __global int* d_permutation, __global double* d_X, __global double* d_fitness)
{
//int NB_GENES = get_NB_GENES(0); // size of a work group (number of items in a work group); number of genes
// The different ids
    int id_global = get_global_id(0);
    int id_local = get_local_id(0);
    int id_group = get_group_id(0);

// We first copy the values of the first parent to the corresponding child
    d_permutation[id_group*NB_GENES + id_local] = d_permutation_parents[id_group*NB_GENES + id_local];

// We wait for the copy operation to finish
    barrier(CLK_GLOBAL_MEM_FENCE);

// The random number generator
    unsigned int seed = d_operator_probability[id_group];

// The crossover operation
    if(id_local == 0) {
        int parent = rand_int(seed, get_num_groups(0)); // Who will be the second parent
        seed = xorshift32(seed); // We alway update the seed after each random number generation
        while(parent == id_group) {
            parent = rand_int(seed, get_num_groups(0)); // We ensure that we do not pick the same parent again
            seed = xorshift32(seed);
        }
        int cut = rand_int(seed, NB_GENES - 1); // At which position will the cut of the crossover be?
        seed = xorshift32(seed);
        int which_side = rand_int(seed, 2); // which side of the child will come from the second parent? 0: the left; 1: the right
        seed = xorshift32(seed);
        int k = 0; // increment for the position in the child to be updated

// We loop over all the positions of the second parent to check if the update the child with its value if it is not already found in the other side in the child
        for(int i = 0; i < NB_GENES; i++) {
            int value = d_permutation_parents[parent*NB_GENES + i]; // the value which will be put in the child
            // which_side == 0: we update the left side;  which_side == 1: we update the right side;
            if(!which_side) {
                int j = cut + 1;
                // We check if the value isn't already in the other side of the child
                // j will be an increment of the search position in the child, is in site + 1 <= j < NB_GENES
                while( (d_permutation[id_group*NB_GENES + j] != value) & (j < NB_GENES) )
                    j++;
                // if the value is not found we update the child with the value
                if( j >= NB_GENES) {
                    d_permutation[id_group*NB_GENES + k] =   value;
                    k++;
                }
            }
            else {
                int j = 0;
                // We check if the value isn't already in the other side of the child
                // j will be an increment of the search position in the child, is in 0 <= j < cut
                while( (d_permutation[id_group*NB_GENES + j] != value) & (j <= cut) )
                    j++;
                // if the value is not found we update the child with the value
                if( j > cut) {
                    d_permutation[id_group*NB_GENES + cut + k + 1] =  value;
                    k++;
                }
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for(int i = 0; i < NB_GENES; i++) {
        d_X[id_group*NB_GENES*NB_GENES + id_local*NB_GENES + i] = 0;
    }
    int permut_id = d_permutation[id_group*NB_GENES + id_local];   // Get permutation as a local ID
    d_X[id_group*NB_GENES*NB_GENES + id_local*NB_GENES + permut_id] = 1;

    d_fitness[id_group] = 2.5;
}
