// Fixed parameters: size of the population, the maximum number of generations (first stopping criterion),
// the maximum number of generations where we didn't improve the Best solution (second stopping criterion)

#define __CL_ENABLE_EXCEPTIONS
#define NB_GENES 26

#include "cl.hpp"

#include "util.hpp"
#include "device_picker.hpp"

#include <algorithm>
#include <vector>
#include <array>
#include <random>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ctime>

#include "individual_parallel.hpp"
#include "matrix.hpp"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#include <err_code.h>


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

int main(int argc, char* argv[])
{
	float time;
	clock_t t1, t2;
	t1 = clock();

	if(argc >= 2)
	{
		std::ifstream file_dat;
		file_dat.open(argv[1]);
		if (file_dat.is_open())
		{
			srand(std::time(NULL));
			std::vector<double> F(NB_GENES * NB_GENES); // Flow matrix
			std::vector<double> D(NB_GENES * NB_GENES); // Distance matrix
			int N = open_file_dat(file_dat, F, D);
/*
			std::cout << "Initialization of the Best individual:\n";
			Individual Best;
			generate_Individual(Best, N);; // The Best solution
			print_permutation(Best);
			evaluate_trace(Best, F, D);
			std::cout << "Fitness: " << Best.fitness << std::endl;
			// WARNING: Evaluation isn't included in other functions, so each time
			// the Individual is altered (crossover, mutation, swap etc.) we should ensure
			// the its fitness is updated afterwards*/

			std::vector<int> permutation(NB_GENES); // 1 permutation (after that, pop_size * NB_GENES)
			std::vector<double> X(NB_GENES * NB_GENES);	// 1 permutation matrix
			double fitness = 0;	// fitness of the individual


			// We initialize the population
			// ------------------------------------------------------------------                        
	    	// Create a context and queue                                                                
		    // ------------------------------------------------------------------                        
			cl::Buffer d_F, d_D, d_permutation, d_X;
			try                                                                                          
    		{                                                                                            
		        cl_uint deviceIndex = 1;
		        parseArguments(argc, argv, &deviceIndex);

		        // Get list of devices
		        std::vector<cl::Device> devices;
		        // Insert devices from each platform in devices
		        // & return the size of devices (number of devices)
		        unsigned numDevices = getDeviceList(devices);

		        // Check device index in range
		        if (deviceIndex >= numDevices)
		        {
		            std::cout << "Invalid device index (try '--list')\n";
		            return EXIT_FAILURE;
		        }

		        // Choose my device
		        cl::Device device = devices[deviceIndex];

		        std::string name;
		        getDeviceName(device, name);
		        std::cout << "\nUsing OpenCL device: " << name << "\n";

		        // Creation of the context with the chosen device
		        std::vector<cl::Device> chosen_device;
		        chosen_device.push_back(device);
		        cl::Context context(chosen_device);

		        // Load in kernel source, creating a program object for the context
		        cl::Program program(context, util::loadProgram("generate_individual.cl"), true);
		        // Get the command queue
		        cl::CommandQueue queue(context);

				auto generate_individual = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(program, "generate_individual");

				// ------------------------------------------------------------------
	  	        // Setup the buffers and write them into global memory
		        // ------------------------------------------------------------------
				d_F = cl::Buffer(context, F.begin(), F.end(), true);
				d_D = cl::Buffer(context, D.begin(), D.end(), true);
				d_permutation = cl::Buffer(context, CL_MEM_WRITE_ONLY, NB_GENES * sizeof(int));
				d_X = cl::Buffer(context, CL_MEM_WRITE_ONLY, NB_GENES * NB_GENES * sizeof(double));

				// ------------------------------------------------------------------
		        // OpenCL initialization of the population
		        // ------------------------------------------------------------------

		        // Create the compute kernel from the program
				generate_individual(cl::EnqueueArgs(queue, cl::NDRange(pop_size)), d_F, d_D, d_permutation, d_X);
				
				queue.finish();

				cl::copy(queue, d_permutation, permutation.begin(), permutation.end());
				cl::copy(queue, d_X, X.begin(), X.end());
			}
			catch (cl::Error err) {
				std::cout << "Exception\n";
			    std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
 			}

			for (int i = 0; i < NB_GENES; i++)
				std::cout << permutation[i] << std::endl;

			// for(int i = 0; i < pop_size; i++){
			// 	generate_Individual(Population[i], N);
			// 	evaluate_trace(Population[i], F, D);
			// }

			int generation = 0; // Number of generations
			int no_improvement = 0; // Number of generations since the last time Best was updated
			int best_generation = generation; // In which generation did we find the Best solution?

			// Stopping criteria:
			// 1. We reach the maximum number of generations OR
			// 2. There have been a certain number of generations we haven't updated the Best solution
			/*while( (generation < nb_gen) && (no_improvement < no_improvenment_max) )
			{
				generation++;
				no_improvement++;
				std::cout<<"==============================\n";
				std::cout<<"Generation: "<<generation<<"\n";

				for(int i = 0; i < pop_size; i++){
					double operator_probability;
					operator_probability = rand()/RAND_MAX; // Uniformly select a random real in [0, 1]

					// 1. Crossover with a 60% probability
					if(operator_probability < 0.6){
						// We randomly select an individual from the population so it becomes the second Parent; it has to be different from from the first individual so we randomly select from pop_size - 1 individual
						int ind = rand()%(pop_size - 1);
						if(i <= ind) ind++; // We ensure we do not pick the first Parent
						Individual Offspring;
						// There is a possibility to get rid of the Offspring individual by just using the result vecteur in crossover
						crossover(Population[i], Population[ind], Offspring);
						evaluate_trace(Offspring, F, D);
						// We keep the best individual
						if( Offspring.fitness <= Population[i].fitness)
							copy(Population[i], Offspring);
					}

					operator_probability = rand()/RAND_MAX; // Uniformly select a random real in [0, 1]
					// 2. Mutation with a 10% probability
					if(operator_probability < 0.1)
						mutate(Population[i]);

					// 3. Transposition? Check that, wasn't implemented
					evaluate_trace(Population[i], F, D);
					// The infamous 2-opt heuristic to be applied on our individual
					heuristic_2opt(Population[i], F, D);

					// We update the Best solution if we find a better individual
					if(Population[i].fitness < Best.fitness){
						std::cout<<"---------------- A new Best found\n";
						std::cout<<"Before: "<< Best.fitness;
						copy(Best, Population[i]);
						std::cout << " and after: " << Population[i].fitness << "\n";
						best_generation = generation;
						// We reset the no_improvement iterator
						no_improvement = 0;
					}

				}
			}*/


			t2= clock();
			time= (float)(t2-t1)/CLOCKS_PER_SEC;
/*
			std::cout<<"======================================== Terminated ======================================\n";
			std::cout<<"Best solution found:\n";
			print_permutation(Best);
			std::cout<<"Fitness: " << Best.fitness << "\nExecution time: " << time << " s\nGeneration: " << best_generation << "\n";

			if(argc >= 3){
				std::ifstream file_soln;
				file_soln.open(argv[2]);
				if (file_soln.is_open()){
					double Optimal_Value;
					int* optimal_permutation = (int *) calloc(N, sizeof(int));
					int Nsol = open_file_soln(file_soln, Optimal_Value, optimal_permutation);
					if(Nsol != N){
						std::cout<<"The size of the problem in the solution file is different from the data file, you specified the wrong solution file."<<std::endl;
					}
					else{
						std::cout<<"======================================= Known optimal solution ============================\n";
						Individual Optimal;
						//generate_Individual_noRandom(Optimal, optimal_permutation, Nsol);
						Optimal.N = Nsol;
						for(int i = 0; i < Optimal.N; i++)
							Optimal.permutation[i] = optimal_permutation[i];
						construct_matrix(Optimal);

						evaluate_trace(Optimal, F, D);
						std::cout<<"Known optimal solution:\n";
						print_permutation(Best);
						std::cout << "Optimal value (in the file): " << Optimal_Value << "\n";
						std::cout << "Optimal value (found by the program): " << Best.fitness << "\n";
					}
					std::free(optimal_permutation);
				}
			}*/
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
