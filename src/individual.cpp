#include <vector>
#include <array>
#include <algorithm>
#include <random>

#include <iostream>
#include <fstream>

#include "individual.hpp"
#include "matrix.hpp"


// Constructor that uses random permutation
// n: size of the problem; how many locations are there?
Individual::Individual(int n, std::random_device &r)
{
	std::mt19937 g(r());
	// We fix the size N
	N = n;
	permutation.reserve(N); 
	// We first construct the vctor of all integers
	for(int i = 0; i < N; i++)
		permutation.push_back(i);
	std::shuffle(permutation.begin(), permutation.end(), g); // We randomly permutate the positions
	construct_matrix(); // For the matrix
	fitness = 0; // We initialize the fitness
}


// Constructor that uses predefined permutation
// p: permutation vector
Individual::Individual(std::vector<int> p)
{
	N = p.size(); // We set the size
	permutation = p;
	construct_matrix();
	fitness = 0;	// Should be initialized anyway, but 0?
}


// Build the permutation matrix X from the permutation
void Individual::construct_matrix()
{
	X = std::vector<double>(N*N, 0);	// We fill the matrix with zeros at first

	int j;
	// We update the components that will be 1
	for(int i = 0; i < N; i++){
		j = permutation[i];
		X[i*N + j] = 1;
	}
}


// WARNING: Evaluation isn't included in other functions, so each time
// the Individual is altered (crossover, mutation, swap etc.) we should ensure
// the its fitness is updated afterwards
// Evaluate the objective function
void Individual::evaluate_trace(const std::vector<double> &F, const std::vector<double> &D)
{
	std::vector<double> A; // Will save F*X
	std::vector<double> B; // Will save X*D
	std::vector<double> C; // Will save Dt*Xt

	seq_mat_mul_sdot(N, F, X, A); // A = F*X
	seq_mat_mul_sdot(N, X, D, B); // B = X*D
	mat_transpose(N, B, C); // C = (X*D)t
	seq_mat_mul_sdot(N, A, C, B); // B = A*C

	fitness = mat_trace(N, B); // The value of the objective function
}


// Evaluate the objective function
void Individual::evaluate_original(const std::vector<double> &F, const std::vector<double> &D)
{
	double tmp = 0;
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			tmp += F[i*N + j]*D[permutation[i]*N + permutation[j]];
		}
	}
	fitness = tmp;
}


// Mutation operator: randomly swap two positions
void Individual::mutate(std::random_device &r)
{
	std::mt19937 g(r());
	std::uniform_int_distribution<> distrib1(0, N - 1); // Uniform random distribution for the first position
	std::uniform_int_distribution<> distrib2(0, N - 2); // Uniform random distribution for the second position

	// We get the two positions randomly
	int position1 = distrib1(g);
	int position2 = distrib2(g);
	// We make sure they are not the same and simulate a sampling over {0, ..., position1-1} U {position1+1, ..., N-1}
	if (position2 >= position1)
		position2 = position2 + 1;
	// The swapping of the positions in the permutation
	int buffer = permutation[position1];
	permutation[position1] = permutation[position2];
	permutation[position2] = buffer;

	// We rearrange the matrix
	X[position1*N + permutation[position2]] = 0;
	X[position2*N + permutation[position1]] = 0;

	X[position1*N + permutation[position1]] = 1;
	X[position2*N + permutation[position2]] = 1;
}


// Crossover operator between the Individual and a second one; produces an offspring. One Point Crossover (OPX)
Individual Individual::crossover(const Individual& Individual2, std::random_device &r)
{
	std::mt19937 g(r());
	std::uniform_int_distribution<> distrib(0, N - 2);// distribution of the cross, cannot be the last element, else there is no crossover
	std::uniform_int_distribution<> binary(0,1); // which Individual will be the first parent?

	int site = distrib(g); // The crossing position in the permutation vector
	int bin = binary(g); // Which Individual will be first?
	std::vector<int> offspring; // The resulting permutation vector

	// The Parents' permutation vectors
	std::vector<int> Parent1;
	std::vector<int> Parent2;
	// Assignement of the Parents
	if(bin == 0){
		Parent1 = permutation;
		Parent2 = Individual2.permutation;
	}
	else{
		Parent1 = Individual2.permutation;
		Parent2 = permutation;
	}

	// The first half of the offspring will come from the first parent
	for(int i = 0; i <= site; i++){
		offspring.push_back(Parent1[i]);
	}

	// The second half will be completed by the second parent
	for(int i = 0; i < N; i++)
		// We check that the value isn't already in the offspring vector
		if( std::find(offspring.begin(), offspring.end(), Parent2[i]) == std::end(offspring) )
			offspring.push_back(Parent2[i]);

	Individual Offspring(offspring); // We create an new individual from the resulting permutation vector
	return Offspring;
}


// Swap two predefined positions to get a new permutation vector
void Individual::swap(int i, int j, std::vector<int> &swap_perm)
{
	// We empty the resulting vector
	swap_perm.clear();
	swap_perm.reserve(N);
	// We fill it t with the values of the permutation vector
	for(int k = 0; k < N; k++)
		swap_perm.push_back(permutation[k]);
	// We permutate the two positions
	swap_perm[i] = permutation[j];
	swap_perm[j] = permutation[i];
}


// 2-opt heuristic: We keep swapping all possible combinations of two positions
// until we exhaust all possibilities, while keeping track of the Best solution found
void Individual::heuristic_2opt(const std::vector<double> &D, const std::vector<double> &F)
{
	// We create a Best individual to keep track of the best solution
	Individual Best(permutation);
	Best.evaluate_trace(F, D);	// Evaluate fitness
	// For each position i 
	for(int i = 0; i < N; i++){
		// We swap it with another position j
		for(int j = i + 1; j < N; j++){
			std::vector<int> v; // This vector will contain the swapping
			Best.swap(i, j, v);	// v is set with the swapping
			Individual Swapped(v); // Resulting individual
			Swapped.evaluate_trace(F, D);	// Evaluate fitness
			// We update Best whenever we find a better solution
			if(Swapped.fitness < Best.fitness){
				Best = Swapped;	// The affectation is shallow, maybe create an assignment operator in the class
				// We reset the positions
				i = 0;
				j = i + 1;
			}
		}
	}
	// Updating the variables with those of Best
	permutation = Best.permutation;
	X = Best.X;
	fitness = Best.fitness;
}


// Print the permutation vector
void Individual::print_permutation()
{
	for(int i = 0; i < N; i++)
		std::cout << permutation[i] << "  ";
	std::cout << std::endl;
}


// Print the permutation matrix
void Individual::print_matrix()
{
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++)
			std::cout << X[N*i + j] << " ";
		std::cout << std::endl;
	}
}
