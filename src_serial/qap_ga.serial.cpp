// Fixed parameters: size of the population, the maximum number of generations (first stopping criterion),
// the maximum number of generations where we didn't improve the Best solution (second stopping criterion)

#include <algorithm>
#include <vector>
#include <array>
#include <random>

#include <iostream>
#include <fstream>

#include "individual.serial.hpp"
#include "matrix.serial.hpp"

#define pop_size 1000
#define nb_gen 250
#define no_improvenment_max 25
#define NB_TESTS 100


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
	Solution.resize(N);
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
	t1 = clock();

#ifdef TESTS
	int nb_tests = NB_TESTS;
	std::vector<float> exec_times(nb_tests, -1);
	std::vector<double> errors(nb_tests, -1);

	for (int index = 0; index < nb_tests; index++) {
		t1 = clock();
#endif
	std::random_device r;
	if(argc >= 2)
	{
		std::ifstream file_dat;
		file_dat.open(argv[1]);
		if (file_dat.is_open()){
			std::uniform_real_distribution<> U_distr(0.0, 1.0); // Real uniform distribution for deciding whether or not to perform an operator
			std::vector<double> F; // Flow matrix
			std::vector<double> D; // Distance matrix
			int N = open_file_dat(file_dat, F, D);

			std::vector<Individual> Population(pop_size, Individual(N, r)); // The population 
			std::vector<Individual> Parents(pop_size, Individual(N, r)); // The population 

			Individual Best(N, r); // The Best solution
			// WARNING: Evaluation isn't included in other functions, so each time
			// the Individual is altered (crossover, mutation, swap etc.) we should ensure
			// the its fitness is updated afterwards
			Best.evaluate_trace(F, D);

#ifndef TESTS
			std::cout << "Initialization of the Best individual:\n";
			Best.print_permutation();
			std::cout << "Fitness: " << Best.fitness << std::endl;
#endif
			// We initialize the population
			for(int i = 0; i < Population.size(); i++){
				Individual buffer(N, r);
				Population[i] = buffer;
				Population[i].evaluate_trace(F, D);
				Parents[i] = Population[i];
			}

			int generation = 0; // Number of generations
			int no_improvement = 0; // Number of generations since the last time Best was updated
			int best_generation = generation; // In which generation did we find the Best solution?

			// Stopping criteria:
			// 1. We reach the maximum number of generations OR
			// 2. There have been a certain number of generations we haven't updated the Best solution
			while( (generation < nb_gen) && (no_improvement < no_improvenment_max) )
			{
				generation++;
				no_improvement++;
#ifndef TESTS
				std::cout<<"==============================\n";
				std::cout<<"Generation: "<<generation<<"\n";
#endif
				// We make a copy of the population; it will represent the parents and we need it so we do not change 
				// the individuals/parents during the process
				for(int i = 0; i < Population.size(); i++){
					Parents[i] = Population[i];
				}

				for(int i = 0; i < Population.size(); i++){
					std::mt19937 g(r());
					double operator_probability;
					operator_probability = U_distr(g); // Uniformly select a random real in [0, 1]

					// 1. Crossover 
					// We randomly select an individual from the previous population so it becomes the second Parent; the first parent will have the same index
					std::uniform_int_distribution<> distrib(0, Population.size() - 2); 
					int ind = distrib(g);
					if(i <= ind) ind++; // We ensure we do not pick the first Parent
					Individual Offspring = Parents[i].crossover(Parents[ind], r);

					operator_probability = U_distr(g); // Uniformly select a random real in [0, 1]
					// 2. Mutation with a 50% probability
					if(operator_probability < 0.5)
						Offspring.mutate(r);
					Offspring.evaluate_trace(F, D);

					// We keep the best individual between the offspring and its parent with a probability of 90%
					operator_probability = U_distr(g); // Uniformly select a random real in [0, 1]
					if(operator_probability < 0.9){
						if(Offspring.fitness < Parents[i].fitness){
							Population[i] = Offspring;
							Population[i].evaluate_trace(F, D);
						}
						// no need to copy an individual in population from parents because the former is already a copy of the later
					}
					else{
						if(Offspring.fitness >= Parents[i].fitness){
							Population[i] = Offspring;
							Population[i].evaluate_trace(F, D);
						}
					}

					// We update the Best solution if we find a better individual
					if(Population[i].fitness < Best.fitness){
#ifndef TESTS
						std::cout<<"---------------- A new Best found\n";
						std::cout<<"Before: "<< Best.fitness;
#endif
						Best = Population[i];
#ifndef TESTS
						std::cout << " and after: " << Population[i].fitness << "\n";
#endif
						best_generation = generation;
						// We reset the no_improvement iterator
						no_improvement = 0;
					}

				}
			}

			t2= clock();
			time= (float)(t2-t1)/CLOCKS_PER_SEC;
#ifdef TESTS
			exec_times[index] = time;
#endif

#ifndef TESTS
			std::cout<<"======================================== Terminated ======================================\n";
			std::cout<<"Best solution found:\n";
			Best.print_permutation();
			std::cout<<"Fitness: " << Best.fitness << "\nExecution time: " << time << " s\nGeneration: " << best_generation << "\n";
#endif
			if(argc >= 3){
				std::ifstream file_soln;
				file_soln.open(argv[2]);
				if (file_soln.is_open()){
					std::vector<int> optimal_permutation;
					double Optimal_Value;
					open_file_soln(file_soln, Optimal_Value, optimal_permutation);
#ifndef TESTS
					std::cout<<"======================================= Known optimal solution ============================\n";
#endif
					Individual Optimal(optimal_permutation); // was used to test the evaluate_fitness function; the print_permutation is handy
					Optimal.evaluate_trace(F, D);
#ifndef TESTS
					std::cout<<"Known optimal solution:\n";
					Optimal.print_permutation();
					std::cout << "Optimal value (in the file): " << Optimal_Value << "\n";
					std::cout << "Best solution found by the genetic algorithm: " << Best.fitness << "\n";
#else
					errors[index] = abs(Best.fitness - Optimal_Value) / Optimal_Value;
#endif
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
#ifdef TESTS
	}
	float average_time = accumulate(exec_times.begin(), exec_times.end(), 0.0) / nb_tests;
	double average_err = accumulate(errors.begin(), errors.end(), 0.0) / nb_tests;

	std::cout << "Number of iterations in the test: " << nb_tests << std::endl;
	std::cout << "Average execution time: " << average_time << " sec" << std::endl;
	std::cout << "Average relative error between known and found optimal fitness: " << average_err << std::endl;
#endif

	return 0;
}
