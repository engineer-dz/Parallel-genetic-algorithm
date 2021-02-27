/*------------------------------------------------------------------------------
**
** Kernel:  generation
**
** Purpose: simulate a generation of the genetic algorithm
**
** Input:
**
** Output:
**
*/

// Functions
uint xorshift32(uint seed);
int rand_int(uint random, int n);
float rand_float(uint random);
void compute_fitness(__global double *d_F, __global double *d_D, __global double* d_A, __global double* d_B, __global double* d_C, __global double *d_X, __global double *d_fitness);
void update_X(__global double* d_X, __global int* d_permutation);

// Kernel
__kernel void generation(__global double* d_F, __global double* d_D, __global double* d_A, __global double* d_B, __global double* d_C, __global unsigned int* d_operator_probability, __global int* d_permutation_parents, __global double* d_X_parents, __global double* d_fitness_parents, __global int* d_permutation, __global double* d_X, __global double* d_fitness, __global int* d_permutations_2_opt, __global double* d_best_2_opt)
{
    int size_work_group = get_local_size(0); // size of a work group (number of items in a work group); number of genes
// The different ids
    int id_global = get_global_id(0);
    int id_local = get_local_id(0);
    int id_group = get_group_id(0);

// We first copy the values of the first parent to the corresponding child
    d_permutation[id_group*size_work_group + id_local] = d_permutation_parents[id_group*size_work_group + id_local];

// We wait for the copy operation to finish
    barrier(CLK_GLOBAL_MEM_FENCE);

// The seed of the random number generator; need to be updated after each number generation
    unsigned int seed = d_operator_probability[id_group];

// The crossover operation; the first item will take care of it
    if(id_local == 0) {
        int parent = rand_int(seed, get_num_groups(0)); // Who will be the second parent?
        seed = xorshift32(seed); // We alway update the seed after each random number generation
        while(parent == id_group) {
            parent = rand_int(seed, get_num_groups(0)); // We ensure that we do not pick the same parent again
            seed = xorshift32(seed);
        }
        int cut = rand_int(seed, size_work_group - 1); // At which position will the cut of the crossover be?
        seed = xorshift32(seed);
        int which_side = rand_int(seed, 2); // which side of the child will come from the second parent? 0: the left; 1: the right
        seed = xorshift32(seed);
        int k = 0; // increment for the position in the child to be updated

// We loop over all the positions of the second parent to find those that are not already in the other side of the cut in the child
        for(int i = 0; i < size_work_group; i++) {
            int value = d_permutation_parents[parent*size_work_group + i]; // the value which will be put in the child
            // which_side == 0: we update the left side;  which_side == 1: we update the right side;
            if(!which_side) {
                int j = cut + 1;
                // We check if the value isn't already in the other side of the child
                // j will be an increment of the search position in the child, is in: site + 1 <= j < size_work_group
                while( (d_permutation[id_group*size_work_group + j] != value) & (j < size_work_group) )
                    j++;
                // if the value is not found we update the child with the value
                if( j >= size_work_group) {
                    d_permutation[id_group*size_work_group + k] =   value;
                    k++;
                }
            }
            else {
                int j = 0;
                // We check if the value isn't already in the other side of the child
                // j will be an increment of the search position in the child, is in 0 <= j < cut
                while( (d_permutation[id_group*size_work_group + j] != value) & (j <= cut) )
                    j++;
                // if the value is not found we update the child with the value
                if( j > cut) {
                    d_permutation[id_group*size_work_group + cut + k + 1] =  value;
                    k++;
                }
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE); // We wait for the crossover to be finished

// The mutation operation; the first item will take care of it
    if(id_local == 0) {
        int mutate_or_not = rand_float(seed);
        seed = xorshift32(seed);
// There is a 50% chance of mutation
        if(mutate_or_not < 0.5) {

            // We get the two positions randomly
            int position1 = rand_int(seed, size_work_group);
            seed = xorshift32(seed);
            int position2 = rand_int(seed, size_work_group - 1);
            seed = xorshift32(seed);
            // We make sure they are not the same and simulate a sampling over {0, ..., position1-1} U {position1+1, ..., N-1}
            if (position2 >= position1)
                position2 = position2 + 1;
            // The swapping of the positions in the permutation
            int buffer = d_permutation[id_group*size_work_group + position1];
            d_permutation[id_group*size_work_group + position1] = d_permutation[id_group*size_work_group + position2];
            d_permutation[id_group*size_work_group + position2] = buffer;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE); // We wait for the mutation to be finished

    update_X(d_X, d_permutation); // We update the permutation matrix
    barrier(CLK_GLOBAL_MEM_FENCE);

    compute_fitness(d_F, d_D, d_A, d_B, d_C, d_X, d_fitness); // We compute the fitness
    barrier(CLK_GLOBAL_MEM_FENCE);
// 2-opt heuristic: We swap all possible combinations of two positions
// until we exhaust all possibilities, while keeping track of the Best solution found
// We fill up the permatations of the 2-opt; there will be permutations of permutations
    for(int j = 0; j < size_work_group; j++) {
        d_permutations_2_opt[id_group*size_work_group + id_local*size_work_group + j] = d_permutation[id_group*size_work_group + j];
    }

{
    // We create a best fitness to keep track of improvements
    d_best_2_opt[id_group*size_work_group + id_local] = d_fitness[id_local*size_work_group];
    double swap_fitness = d_best_2_opt[id_group*size_work_group + id_local];
   int i = id_local; 
    // For each position id_local (referred as i)
    // We swap it with another position j
    for(int j = i + 1; j < size_work_group; j++) {
        int permutation_i = d_permutations_2_opt[id_group*size_work_group + id_local*size_work_group + i];
        int permutation_j = d_permutations_2_opt[id_group*size_work_group + id_local*size_work_group + j];
        swap_fitness = swap_fitness - d_F[i]*d_D[permutation_i] - d_F[j]*d_D[permutation_j] + d_F[i]*d_D[permutation_j] + d_F[j]*d_D[permutation_i];

        // We update the individual whenever we find a better solution
        if(swap_fitness < d_best_2_opt[id_group*size_work_group + id_local]) {
            // We copy the content of the array in the permutation
            d_permutations_2_opt[id_group*size_work_group + id_local*size_work_group + i] = permutation_j;
            d_permutations_2_opt[id_group*size_work_group + id_local*size_work_group + j] = permutation_i;
	    d_best_2_opt[id_group*size_work_group + id_local] = swap_fitness;
        }
    }
}

    if(id_local == 0) {
    int best_permutation = 0;
        for(int i = 0; i < size_work_group ; i++) {
		if(d_best_2_opt[id_group*size_work_group + i] <  d_best_2_opt[id_group*size_work_group + best_permutation]){
			best_permutation = i;
			}
			}
            //We then copy the result back to the population permutation
            for(int i = 0; i < size_work_group; i++) {
                d_permutation[id_group*size_work_group + i] = d_permutations_2_opt[id_group*size_work_group + best_permutation*size_work_group + i];
            }
        }
    barrier(CLK_GLOBAL_MEM_FENCE); // We wait for the mutation to be finished

    update_X(d_X, d_permutation); // We update the permutation matrix
    barrier(CLK_GLOBAL_MEM_FENCE);

    compute_fitness(d_F, d_D, d_A, d_B, d_C, d_X, d_fitness); // We compute the fitness
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

// Generate random real in [0, 1]
float rand_float(uint random)
{
    return (double) random / (double) UINT_MAX;
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
