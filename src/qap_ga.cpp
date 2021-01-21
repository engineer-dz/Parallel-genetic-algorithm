// Fixed parameters: size of the population, the maximum number of generations (first stopping criterion),
// the maximum number of generations where we didn't improve the Best solution (second stopping criterion)

#include <algorithm>
#include <vector>
#include <array>
#include <random>

#include <iostream>
#include <fstream>

#include "individual.hpp"
#include "matrix.hpp"

#define pop_size 100
#define nb_gen 20
#define no_improvenment_max 5


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
int open_file_soln(std::ifstream &file_soln, double &Value, std::vector<int> &Solution)
{
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



int main(int argc, char* argv[])
{
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
				std::cout<<"Problem while opening the data file " << argv[1] << ".\n";
				return 2;
			}
		}
		else{
			std::cout<<"Please specify a data file to open. Usage:\n";
			std::cout<<"./qap_ga.out file.dat\n";
			return 1;
		}

        return 0;
}
