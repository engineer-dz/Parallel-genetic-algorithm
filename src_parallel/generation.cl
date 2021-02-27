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
void compute_fitness(__global double *d_F, __global double *d_D, __global double* d_A, __global double* d_B, __global double* d_C, __global double *d_X, __global double *d_fitness);
void update_X(__global double* d_X, __global int* d_permutation);

// Kernel
__kernel void generation(__global double* d_F, __global double* d_D, __global double* d_A, __global double* d_B, __global double* d_C, __global unsigned int* d_operator_probability, __global int* d_permutation_parents, __global double* d_X_parents, __global double* d_fitness_parents, __global int* d_permutation, __global double* d_X, __global double* d_fitness)
{
    int num_work_groups = get_local_size(0); // size of a work group (number of items in a work group); number of genes
// The different ids
    int id_global = get_global_id(0);
    int id_local = get_local_id(0);
    int id_group = get_group_id(0);

// We first copy the values of the first parent to the corresponding child
    d_permutation[id_group*num_work_groups + id_local] = d_permutation_parents[id_group*num_work_groups + id_local];

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
        int cut = rand_int(seed, num_work_groups - 1); // At which position will the cut of the crossover be?
        seed = xorshift32(seed);
        int which_side = rand_int(seed, 2); // which side of the child will come from the second parent? 0: the left; 1: the right
        seed = xorshift32(seed);
        int k = 0; // increment for the position in the child to be updated

// We loop over all the positions of the second parent to find those that are not already in the other side of the cut in the child
        for(int i = 0; i < num_work_groups; i++) {
            int value = d_permutation_parents[parent*num_work_groups + i]; // the value which will be put in the child
            // which_side == 0: we update the left side;  which_side == 1: we update the right side;
            if(!which_side) {
                int j = cut + 1;
                // We check if the value isn't already in the other side of the child
                // j will be an increment of the search position in the child, is in: site + 1 <= j < num_work_groups
                while( (d_permutation[id_group*num_work_groups + j] != value) & (j < num_work_groups) )
                    j++;
                // if the value is not found we update the child with the value
                if( j >= num_work_groups) {
                    d_permutation[id_group*num_work_groups + k] =   value;
                    k++;
                }
            }
            else {
                int j = 0;
                // We check if the value isn't already in the other side of the child
                // j will be an increment of the search position in the child, is in 0 <= j < cut
                while( (d_permutation[id_group*num_work_groups + j] != value) & (j <= cut) )
                    j++;
                // if the value is not found we update the child with the value
                if( j > cut) {
                    d_permutation[id_group*num_work_groups + cut + k + 1] =  value;
                    k++;
                }
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE); // We wait for the crossover to be finished
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

// A function that updates the permutation matrix X using the permutation array
void update_X(__global double* d_X, __global int* d_permutation) {
    int num_work_groups = get_local_size(0); // size of a work group (number of items in a work group); number of genes
// The different ids
    int id_global = get_global_id(0);
    int id_local = get_local_id(0);
    int id_group = get_group_id(0);

    // We fill the matrices with 0's
    for(int i = 0; i < num_work_groups; i++) {
        d_X[id_group*num_work_groups*num_work_groups + id_local*num_work_groups + i] = 0;
    }
    int permut_id = d_permutation[id_group*num_work_groups + id_local];   // Get permutation as a local ID
    d_X[id_group*num_work_groups*num_work_groups + id_local*num_work_groups + permut_id] = 1; // We fill the positions corresponding to the permutation with 1's
}

// A function that computes the fitnesses
void compute_fitness(__global double *d_F, __global double *d_D, __global double* d_A, __global double* d_B, __global double* d_C, __global double *d_X, __global double *d_fitness) {
    int num_work_groups = get_local_size(0); // size of a work group (number of items in a work group); number of genes
// The different ids
    int id_global = get_global_id(0);
    int id_local = get_local_id(0);
    int id_group = get_group_id(0);

    double tmp; // Stores intermediate values for matrix multiplication
    // A = F*X
    for (int j = 0; j < num_work_groups; j++) {
        tmp = 0.0f;
        for (int k = 0; k < num_work_groups; k++) {
            tmp += d_F[id_local*num_work_groups+k] * d_X[id_group*num_work_groups*num_work_groups + k*num_work_groups+j];
        }
        d_A[id_group*num_work_groups*num_work_groups + id_local*num_work_groups+j] = tmp;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // B = X*D
    for (int j = 0; j < num_work_groups; j++) {
        tmp = 0.0f;
        for (int k = 0; k < num_work_groups; k++) {
            tmp += d_X[id_group*num_work_groups*num_work_groups + id_local*num_work_groups+k] * d_D[k*num_work_groups+j];
        }
        d_B[id_group*num_work_groups*num_work_groups + id_local*num_work_groups+j] = tmp;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // C = (X*D)transpose
    for(int j = 0; j < num_work_groups; j++)
        d_C[id_group*num_work_groups*num_work_groups + id_local*num_work_groups + j] = d_B[id_group*num_work_groups*num_work_groups + j*num_work_groups + id_local];

    barrier(CLK_GLOBAL_MEM_FENCE);

    // B = A*C
    for (int j = 0; j < num_work_groups; j++) {
        tmp = 0.0f;
        for (int k = 0; k < num_work_groups; k++) {
            tmp += d_A[id_group*num_work_groups*num_work_groups + id_local*num_work_groups+k] * d_C[id_group*num_work_groups*num_work_groups + k*num_work_groups+j];
        }
        d_B[id_group*num_work_groups*num_work_groups + id_local*num_work_groups+j] = tmp;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // fitness = trace(B)
    if(id_local == 0) {
        double trace = 0;
        for(int i = 0; i < num_work_groups; i++) {
            trace += d_B[id_group*num_work_groups*num_work_groups + i*num_work_groups + i];
        }
        d_fitness[id_group] = trace;
    }
}
