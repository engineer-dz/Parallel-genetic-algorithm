#include <vector>
#include <array>
#include <algorithm>
#include <random>

#include <iostream>
#include <fstream>
#include <cstring>

#include "individual_parallel.hpp"
#include "matrix.hpp"


// Function that uses random permutation to create an Individual
// n: size of the problem; how many locations are there?
void generate_Individual(Individual &I, int n)
{
	// We fix the size N
	I.N = n;
	// We first construct the vector of all integers
	for(int i=0 ; i<n ;i++){
		I.permutation[i]=i;
	}

	// shuffle
	for (int i = n-1; i >= 0; i--){
		//generate a random number [0, n-1]
		int j = rand() % (i+1);

		//swap the last element with element at random index
		int temp = I.permutation[i];
		I.permutation[i] = I.permutation[j];
		I.permutation[j] = temp;
	}
	construct_matrix(I);
	I.fitness = 0;
}

// Function that uses predefined permutation to create an Individual
// p: permutation vector
void generate_Individual_noRandom(Individual &I, int *p, int n)
{
	I.N = n;
	for(int i = 0; i < I.N; i++)
		I.permutation[i] = p[i];
	construct_matrix(I);
	I.fitness = 0;
}

// Build the permutation matrix X from the permutation vector of I
void construct_matrix(Individual &I)
{
	for(int i = 0; i < I.N; i++){
		for(int j = 0; j < I.N; j++){
			I.X[i*I.N + j] = 0;
		}
	}

	int j;
	// We update the components that will be 1
	for(int i = 0; i < I.N; i++){
		j = I.permutation[i];
		I.X[i*I.N + j] = 1;
	}
}


// WARNING: Evaluation isn't included in other functions, so each time
// the Individual is altered (crossover, mutation, swap etc.) we should ensure
// the its fitness is updated afterwards
// Evaluate the objective function
void evaluate_trace(Individual &I, const std::vector<double> &F, const std::vector<double> &D)
{
	// Do we change that to double*? I think yes
	std::vector<double> A; // Will save F*X
	std::vector<double> B; // Will save X*D
	std::vector<double> C; // Will save Dt*Xt

	// We empty the vector
	A.clear();
	A.reserve(I.N*I.N);
	B.clear();
	B.reserve(I.N*I.N);
	C.clear();
	C.reserve(I.N*I.N);

	//seq_mat_mul_sdot(I.N, F, I.X, A); // A = F*X
	double tmp = 0;

	for (int i = 0; i < I.N; i++) {
		for (int j = 0; j < I.N; j++) {
			tmp = 0.0f;
			for (int k = 0; k < I.N; k++) {
				tmp += F[i*I.N+k] * I.X[k*I.N+j];
			}
			A[i*I.N+j] = tmp;
		}
	}

	tmp = 0;

	//seq_mat_mul_sdot(I.N, I.X, D, B); // B = X*D
	for (int i = 0; i < I.N; i++) {
		for (int j = 0; j < I.N; j++) {
			tmp = 0.0f;
			for (int k = 0; k < I.N; k++) {
				tmp += I.X[i*I.N+k] * D[k*I.N+j];
			}
			B[i*I.N+j] = tmp;
		}
	}

	mat_transpose(I.N, B, C); // C = (X*D)t
	seq_mat_mul_sdot(I.N, A, C, B); // B = A*C

	I.fitness = mat_trace(I.N, B); // The value of the objective function
}

// Evaluate the objective function using the original formula, and a permutatio vector
double evaluate_original(int* p, const int &n, const std::vector<double> &F, const std::vector<double> &D)
{
	double tmp = 0;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			tmp += F[i*n + j]*D[p[i]*n + p[j]];
		}
	}
	return tmp;
}

// Mutation operator: randomly swap two positions
void mutate(Individual &I)
{
	// We get the two positions randomly
	int position1 = rand()%(I.N);
	int position2 = rand()%(I.N - 1);
	// We make sure they are not the same and simulate a sampling over {0, ..., position1-1} U {position1+1, ..., N-1}
	if (position2 >= position1)
		position2 = position2 + 1;
	// The swapping of the positions in the permutation
	int buffer = I.permutation[position1];
	I.permutation[position1] = I.permutation[position2];
	I.permutation[position2] = buffer;

	// We rearrange the matrix
	I.X[position1*I.N + I.permutation[position2]] = 0;
	I.X[position2*I.N + I.permutation[position1]] = 0;

	I.X[position1*I.N + I.permutation[position1]] = 1;
	I.X[position2*I.N + I.permutation[position2]] = 1;
}

// Crossover operator between two individuals; produces an Offspring. One Point Crossover (OPX) method.
void crossover(const Individual& Individual1, const Individual& Individual2, Individual &Offspring)
{
	int site = rand()%(Individual1.N - 1); // The crossing position in the permutation vector
	int bin = rand()%2; // Which Individual will be first?

	// Assignement of the Parents
	if(bin == 0){
		// The first half of the offspring will come from the first parent
		for(int i = 0; i <= site; i++){
			Offspring.permutation[i] = Individual1.permutation[i];
		}

		// The second half will be completed by the second parent
		for(int i = 0; i < Individual1.N; i++)
			// We check that the value isn't already in the offspring vector
			if( std::find(Offspring.permutation + Offspring.N, Offspring.permutation + Offspring.N, Individual2.permutation[i]) != Offspring.permutation + Offspring.N )
				Offspring.permutation[i] = Individual2.permutation[i];
	}
	else{
		// The first half of the offspring will come from the second parent
		for(int i = 0; i <= site; i++){
			Offspring.permutation[i] = Individual2.permutation[i];
		}

		// The second half will be completed by the first parent
		for(int i = 0; i < Individual1.N; i++)
			// We check that the value isn't already in the offspring vector
			if( std::find(Offspring.permutation + Offspring.N, Offspring.permutation + Offspring.N, Individual1.permutation[i]) != Offspring.permutation + Offspring.N )
				Offspring.permutation[i] = Individual1.permutation[i];
	}
	construct_matrix(Offspring);
}

// 2-opt heuristic: We keep swapping all possible combinations of two positions
// until we exhaust all possibilities, while keeping track of the Best solution found
void heuristic_2opt(Individual &I, const std::vector<double> &D, const std::vector<double> &F)
{
	// We create a Best individual to keep track of the best solution
	// For each position i 
	for(int i = 0; i < I.N; i++){
		// We swap it with another position j
		for(int j = i + 1; j < I.N; j++){
			int *v = (int *) calloc(I.N, sizeof(int));; // This vector will contain the swapping

			// v is set with the swapping
			for(int k = 0; k < I.N; k++){
				v[k] = I.permutation[k];
			}

			// We permutate the two positions
			v[i] = I.permutation[j];
			v[j] = I.permutation[i];

			// We update the individual whenever we find a better solution
			if(evaluate_original(v, I.N, F, D) < I.fitness){
				// We copy the content of the array in the permutation
				for(int k = 0; k < I.N; k++)
					I.permutation[k] = v[k];
				construct_matrix(I);
				evaluate_trace(I, F, D);
				// We reset the positions
				i = 0;
				j = i + 1;
			}
			std::free(v);
		}
	}
}

// Print the permutation vector
void print_permutation(const Individual &I)
{
	for(int i = 0; i < I.N; i++)
		std::cout << I.permutation[i] << "  ";
	std::cout << std::endl;
}

// Print the permutation matrix
void print_matrix(const Individual &I)
{
	for(int i = 0; i < I.N; i++){
		for(int j = 0; j < I.N; j++)
			std::cout << I.X[I.N*i + j] << " ";
		std::cout << std::endl;
	}
}

// Copy the values from one individual to the other
void copy(Individual &Dest, const Individual &Source){
	for(int i = 0; i < Source.N; i++){
		Dest.permutation[i] = Source.permutation[i];
		for(int j = 0; j < Source.N; j ++){
			Dest.X[i*Source.N + j] = Source.X[i*Source.N + j];
		}
	}
	Dest.fitness = Source.fitness;
}

// Function to open a data file of the qaplib 
// https://www.opt.math.tugraz.at/qaplib/inst.html
// Flow will contain the flow matrix and Distance the distance matrix, 
// the function returns N the size of the problem
int open_file_dat(std::ifstream &file_dat, std::vector<double> &Flow, std::vector<double> &Distance)
{
	int N;
	file_dat >> N;
	// We empty the matrices
	Flow.clear();
	Flow.reserve(N*N);
	Distance.clear();
	Distance.reserve(N*N);

	// The first matrix is the Flow matrix
	for (int i = 0; i < N*N; i++){
		double tmp;
		file_dat >> tmp;
		Flow.push_back(tmp);
	}

	// The second matrix is the Distance matrix
	for (int i = 0; i < N*N; i++){
		double tmp;
		file_dat >> tmp;
		Distance.push_back(tmp);
	}

	return N;
}


// Function to open a solution file of the qaplib 
// https://www.opt.math.tugraz.at/qaplib/inst.html
// Value contains the optimal value of the loss function and Solution will contain the optimal permutation 
// the function returns N the size of the problem
int open_file_soln(std::ifstream &file_soln, double &Value, int *Solution)
{
	int N;
	file_soln >> N; 
	file_soln >> Value;
	// We empty the vector
	for(int i = 0; i < N; i++){
		int buffer;
		file_soln >> buffer;
		// We substract one so it can confornm to the C indexing 
		Solution[i] = buffer - 1;
	}

	return N;
}

// We test if the GPU produced individuals are OK (not empty for example) by printing a sample of the population
void printing_test(std::vector<int> permutation, std::vector<double> X, std::vector<double> fitness, int sample_size){
	std::cout << "Affichage des résultats" << std::endl;
	std::cout << "==================================================="  << std::endl;
	for (int s = 0; s < sample_size; s++) {
		int ind = rand()%pop_size; 
		std::cout << "Individual " << ind << std::endl;
		std::cout << "Permutation: " << std::endl;
		for (int i = 0; i < NB_GENES; i++)
			std::cout << permutation[ind*NB_GENES + i] << " ";
		std::cout << std::endl;

		std::cout << "X: " <<  std::endl;
		for (int i = 0; i < NB_GENES; i++) {
			for (int j = 0; j < NB_GENES; j++) {
				std::cout << X[ind*NB_GENES*NB_GENES + i * NB_GENES + j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "fitness: " << fitness[ind] << std::endl;
		std::cout << "==================================================="  << std::endl;
	}
}

// We test if the GPU produced fitness is the same as the one computed on an individual I using the function evaluate_trace, tests on a random sample
void fitness_test(std::vector<double> F, std::vector<double> D, std::vector<int> permutation, std::vector<double> X, std::vector<double> fitness, int sample_size){
	std::cout << "Affichage des fitness" << std::endl;
	std::cout << "==================================================="  << std::endl;
	for (int s = 0; s < sample_size; s++) {
		int ind = rand()%pop_size;
		std::cout << "Individual " << ind << std::endl;
		Individual I;
		I.N = NB_GENES;
		for (int i = 0; i < NB_GENES; i++)
			I.permutation[i] = permutation[ind*NB_GENES + i];
		for (int i = 0; i < NB_GENES * NB_GENES; i++)
			I.X[i] = X[ind*NB_GENES*NB_GENES + i];
		//I.fitness = fitness[j];
		evaluate_trace(I, F, D);
		std::cout << "GPU fitness: " << fitness[ind] << std::endl;
		std::cout << "evaluated fitness: " << I.fitness << std::endl;
		std::cout << "==================================================="  << std::endl;
	}
}
