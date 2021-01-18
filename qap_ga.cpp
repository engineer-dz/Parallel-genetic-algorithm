// Fixed parameters: size of the population, the maximum number of generations (first stopping criterion), the maximum number of generations where we didn't improve the Best solution (second stopping criterion)
#define pop_size 100
#define nb_gen 20
#define no_improvenment_max 5
#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <random>
#include <fstream>

class Individual {
	int N; // size of the permutation vector
	std::vector<int> permutation; // permutation vector
	std::vector<double> X; // permutation matrix representation
	public:
	double fitness; // value of the objective function for this solution

	Individual(int, std::random_device&); // Constructor that uses random permutation
	Individual(std::vector<int>); // Constructor that uses predefined permutation
	void construct_matrix(); // Build the permutation matrix X from the permutation
	// WARNING: Evaluation isn't included in other functions, so each time
	// the Individual is altered (crossover, mutation, swap etc.) we should ensure
	// the its fitness is updated afterwards
	void evaluate_trace(std::vector<double>, std::vector<double>); // Evaluate the objective function
	void evaluate_original(std::vector<double>, std::vector<double>); // Evaluate the objective function
	void mutate(std::random_device&); // Mutation operator: randomly swap two positions
	void swap(int, int, std::vector<int> &); // Swap two predefined positions
	void heuristic_2opt(std::vector<double>, std::vector<double>); // 2-opt heuristic
	Individual crossover(Individual&, std::random_device&); // Crossover with another individual to get an Offspring
	void print_permutation(); // Print the permutation vector
	void print_matrix(); // Print the permutation matrix
};

// Compute the transpose of a matrix
void mat_transpose(int N, std::vector<double> &A, std::vector<double> &At){
	int i, j;
	// We empty the vector
	At.clear();
	At.reserve(N*N);

	for(i = 0; i < N; i++)
		for(j = 0; j < N; j++)
			At[i*N + j] = A[j*N + i];
}

// Compute the trace of a matrix
double mat_trace(int N, std::vector<double> &A){
	int i;
	double tmp = 0;
	for(i = 0; i < N; i++)
		tmp += A[i*N + i];
	return tmp;
}

// Compute the product of two matrices A and B
void seq_mat_mul_sdot(int N, std::vector<double> &A, std::vector<double> &B, std::vector<double> &C)
{
	int i, j, k;
	double tmp;
	// We empty the vector
	C.clear();
	C.reserve(N*N);

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			tmp = 0.0f;
			for (k = 0; k < N; k++) {
				tmp += A[i*N+k] * B[k*N+j];
			}
			C[i*N+j] = tmp;
		}
	}
}

// Constructor that uses random permutation
// n: size of the problem; how many locations are there?
Individual::Individual(int n, std::random_device &r){
	std::mt19937 g(r());
	// We fix the size N
	N = n;
	permutation.reserve(N); 
	// We first construct the vctor of all integers
	for(int i = 0; i < N; i++)
		permutation.push_back(i);
	std::shuffle(permutation.begin(), permutation.end(), g); // We randomly permutate the positions
	construct_matrix(); // For the matrix
	fitness = 0; // We initialize the fitness; Should we compute it here?
}

// Constructor that uses predefined permutation
// p: permutation vector
Individual::Individual(std::vector<int> p) {
	N = p.size(); // We set the size
	permutation = p;
	construct_matrix();
}

// Build the permutation matrix X from the permutation
void Individual::construct_matrix() {
	X.reserve(N*N);
	std::fill(X.begin(), X.end(), 0); // We fill the matrix with zeros at first
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			X.push_back(0);

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
void Individual::evaluate_trace(std::vector<double> F, std::vector<double> D){
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
void Individual::evaluate_original(std::vector<double> F, std::vector<double> D){
	double tmp;
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			tmp += F[i*N + j]*D[permutation[i]*N + permutation[j]];
		}
	}
	fitness = tmp;
}

// Mutation operator: randomly swap two positions
void Individual::mutate(std::random_device &r){
	std::mt19937 g(r());
	std::uniform_int_distribution<> distrib1(0, N - 1); // Uniform random distribution for the first position
	std::uniform_int_distribution<> distrib2(0, N - 2); // Uniform random distribution for the second position

	// We get the two positions randomly
	int position1 = distrib1(g);
	int position2 = distrib2(g);
	// We make sure they are not the same
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
Individual Individual::crossover(Individual& Individual2, std::random_device &r){
	std::mt19937 g(r());
	std::uniform_int_distribution<> distrib(0, N - 2);// distribution of the cross
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
		if( std::find(offspring.begin(), offspring.end(), Parent2[i]) == std::end(offspring) ){
			offspring.push_back(Parent2[i]);
		}

	Individual Offspring(offspring); // We create an new individual from the resulting permutation vector
	return Offspring;
}

// Swap two predefined positions to get a new permutation vector
void Individual::swap(int i, int j, std::vector<int> &swap_perm){
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
void Individual::heuristic_2opt(std::vector<double> D, std::vector<double> F){
	// We create a Best individual to keep track of the best solution
	Individual Best(permutation);
	Best.evaluate_trace(F, D);
	// For each position i 
	for(int i = 0; i < N; i++){
		// We permute it with another position j
		for(int j = i + 1; j < N; j++){
			std::vector<int> v; //This vector will contain the swapping
			Best.swap(i, j, v);
			Individual Swapped(v); // Resulting individual
			Swapped.evaluate_trace(F, D);
			// We update Best whenever we find a better solution
			if(Swapped.fitness < Best.fitness){
				Best = Swapped;
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
void Individual::print_permutation(){
	for(int i = 0; i < N; i++)
		std::cout << permutation[i] << "  ";
	std::cout << "\n";
}

// Print the permutation matrix
void Individual::print_matrix(){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++)
			std::cout << X[N*i + j] << " ";
		std::cout << "\n";
	}
}

// Print a matrix
// n is the number of rows and m of columns
void print_matrix(std::vector<double> matrix, int n, int m){
	for(int i = 0; i < n; i++){
		std::cout << "\n";
		for(int j = 0; j < n; j++)
			std::cout << matrix[n*i + j] << " ";
	}
	std::cout << "\n";
}

// Function to open a data file of the qaplib 
// https://www.opt.math.tugraz.at/qaplib/inst.html
// Flow will contain the flow matrix and Distance the distance matrix, 
// the function returns N the size of the problem
int open_file_dat(std::ifstream &file_dat, std::vector<double> &Flow, std::vector<double> &Distance){
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
int open_file_soln(std::ifstream &file_soln, double &Value, std::vector<int> &Solution){
	int N;
	file_soln >> N; 
	file_soln >> Value;
	// We empty the vector
	Solution.clear();
	Solution.reserve(N);
	for(int i = 0; i < N; i++){
		int buffer;
		file_soln >> buffer;
		// We substract one so it can confornm to the C indexing 
		Solution[i] = buffer - 1;
	}

	return N;
}

int main(int argc, char* argv[]){
	float time;
	clock_t t1, t2;
	t1= clock();

	std::random_device r;
	if(argc>= 2)
	{
		std::ifstream file_dat;
		file_dat.open(argv[1]);
		if (file_dat.is_open()){
			std::uniform_real_distribution<> U_distr(0.0, 1.0); // Real uniform distribution for deciding whether or not to perform an operator
			std::vector<double> F; // Flow matrix
			std::vector<double> D; // Distance matrix
			int N = open_file_dat(file_dat, F, D);

			std::vector<Individual *> Population; // The population 
			Population.reserve(pop_size);

			Individual Best(N, r); // The Best solution
	// WARNING: Evaluation isn't included in other functions, so each time
	// the Individual is altered (crossover, mutation, swap etc.) we should ensure
	// the its fitness is updated afterwards
			Best.evaluate_trace(F, D);

			std::cout<<"Initialization of the Best individual:\n";
			Best.print_permutation();
			std::cout<<"Fitness :" << Best.fitness<<"\n";
			// We initialize the population
			for(int i = 0; i < pop_size; i++){
				Population.push_back(new Individual(N, r));
				Population[i]->evaluate_trace(F, D);
			}

			int generation = 0; // Number of generations
			int no_improvement = 0; // Number of generations since the last time Best was updated
			int best_generation = generation; // In which generation did we find the Best solution?

			// Stopping criteria:
			// 1. We reach the maximum number of generations OR
			// 2. There have been a certain number of generations we haven't updated the Best solution
			while( (generation < nb_gen) && (no_improvement < no_improvenment_max) ){
				generation++;
				no_improvement++;
				std::cout<<"==============================\n";
				std::cout<<"Generation: "<<generation<<"\n";
				for(int i = 0; i < Population.size(); i++){
					std::mt19937 g(r());
					double operator_probability;
					operator_probability = U_distr(g); // Uniformly select a random real in [0, 1]
					// 1. Crossover with a 60% probability
					if(operator_probability < 0.6){
						// We randomly select an individual from the population so it becomes the second Parent
						std::uniform_int_distribution<> distrib(0, Population.size() - 2); 
						int ind = distrib(g);
						if(i <= ind) ind++; // We ensure we do not pick the first Parent
						Individual Offspring = Population[i]->crossover(*(Population[ind]), r);
						Offspring.evaluate_trace(F, D);
						// We keep the best individual
						if( Offspring.fitness <= Population[i]->fitness)
							*Population[i] = Offspring;
					}
					operator_probability = U_distr(g); // Uniformly select a random real in [0, 1]
					// 2. Mutation with a 60% probability
					if(operator_probability < 0.1)
						Population[i]->mutate(r);
					// 3. Transposition? Check that, wasn't implemented
					Population[i]->evaluate_trace(F, D);
					// The infamous 2-opt heuristic to be applied on our individual
					Population[i]->heuristic_2opt(F, D);
					//std::cout<<"Value of the Loss function: " << Population[i]->fitness<<"\n";
					// We update the Best solution if we find a better individual
					if(Population[i]->fitness < Best.fitness){
						std::cout<<"---------------- A new Best found\n";
						std::cout<<"Before: "<< Best.fitness;
						Best = *Population[i];
						std::cout << " and after: " << Population[i]->fitness << "\n";
						best_generation = generation;
						// We reset the no_improvement iterator
						no_improvement = 0;
					}

				}
			}
			t2= clock();
			time= (float)(t2-t1)/CLOCKS_PER_SEC;

			std::cout<<"======================================== Terminated ======================================\n";
			std::cout<<"Best solution found:\n";
			Best.print_permutation();
			std::cout<<"Fitness: " << Best.fitness << "\nExecution time: " << time << " s\nGeneration: " << best_generation << "\n";
			if(argc >= 3){
				std::ifstream file_soln;
				file_soln.open(argv[2]);
				if (file_soln.is_open()){
					std::vector<int> optimal_permutation;
					double Optimal_Value;
					open_file_soln(file_soln, Optimal_Value, optimal_permutation);
						std::cout<<"======================================= Known optimal solution ============================\n";
						Individual Optimal(optimal_permutation);
						Optimal.evaluate_trace(F, D);
						std::cout<<"Known optimal solution:\n";
						Best.print_permutation();
						std::cout << "Optimal value (in the file): " << Optimal_Value << "\n";
						std::cout << "Optimal value (computed): " << Optimal.fitness << "\n";
					}
				}

			}
			else{
				std::cout<<"Problem while opening the data file.\n";
				return 2;
			}
		}
		else{
			std::cout<<"Please specify a data file to open.\n";
			return 1;
		}
		return 0;
	}
