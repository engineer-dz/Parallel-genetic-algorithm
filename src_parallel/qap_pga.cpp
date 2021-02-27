
#define __CL_ENABLE_EXCEPTIONS

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


int main(int argc, char* argv[])
{
	// To compute execution time
	float time;
	clock_t t1, t2;
	t1 = clock();

	// At least one argument should be provided by the user, the .dat file
	if(argc >= 2)
	{
		std::ifstream file_dat;
		file_dat.open(argv[1]);
		if (file_dat.is_open())
		{
			srand(std::time(NULL));
			std::vector<double> F(NB_GENES * NB_GENES); // Flow matrix
			std::vector<double> D(NB_GENES * NB_GENES); // Distance matrix
			int N = open_file_dat(file_dat, F, D); // The dimension of the problem provided by the .dat file

			// We ensure that the dimension provided by the file matches the one set in the program (see individual_parallel.hpp)
			if(N == NB_GENES){
				// We create the Best individual, the individual in whom we will keep the best found solution so far 
				std::cout << "Initialization of the Best individual:\n";
				Individual Best;
				generate_Individual(Best, N);; // The Best solution
				print_permutation(Best);
				evaluate_trace(Best, F, D);
				std::cout << "Fitness: " << Best.fitness << std::endl;
				// WARNING: Evaluation isn't included in other functions, so each time
				// the Individual is altered (crossover, mutation, swap etc.) we should ensure
				// the its fitness is updated afterwards

				// The arrays in which we will keep the information about the population
				std::vector<int> permutation(NB_GENES * pop_size); // permutations
				std::vector<double> X(NB_GENES * NB_GENES * pop_size, 0);	// permutation matrices
				std::vector<double> fitness(pop_size);	// fitness of the individuals

				// Initialization of the permutations, this is done in the CPU
				for (int i = 0; i < pop_size; i++) {
					for (int j = 0; j < NB_GENES; j++) {
						permutation[i*NB_GENES + j] = j;
					}
					if (i < pop_size)
						std::random_shuffle(permutation.begin() + i*NB_GENES, permutation.begin() + i*NB_GENES + NB_GENES);
				}

				// We initialize the population in the GPU
				// ------------------------------------------------------------------                        
				// Create a context and queue                                                                
				// ------------------------------------------------------------------                        
				cl::Buffer d_F, d_D, d_A, d_B, d_C, d_permutation, d_X, d_fitness;

				int generation = 0; // Number of generations
				int no_improvement = 0; // Number of generations since the last time Best was updated
				int best_generation = generation; // In which generation did we find the Best solution?
				try                                                                                          
				{                                                                                            
					cl_uint deviceIndex = 0;
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

					auto generate_individual = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(program, "generate_individual");

					// ------------------------------------------------------------------
					// Setup the buffers and write them into global memory
					// ------------------------------------------------------------------
					// The matrices related to the problem, provided by the instance file .dat
					d_F = cl::Buffer(context, F.begin(), F.end(), true);
					d_D = cl::Buffer(context, D.begin(), D.end(), true);
					// Intermediate matrices that will keep information needed to evaluate the fitness of an individual
					d_A = cl::Buffer(context, CL_MEM_READ_WRITE, pop_size * NB_GENES * NB_GENES * sizeof(double));
					d_B = cl::Buffer(context, CL_MEM_READ_WRITE, pop_size * NB_GENES * NB_GENES * sizeof(double));
					d_C = cl::Buffer(context, CL_MEM_READ_WRITE, pop_size * NB_GENES * NB_GENES * sizeof(double));
					// Where information about the population will be stored in the GPU
					d_permutation = cl::Buffer(context, permutation.begin(), permutation.end(), true);
					d_X = cl::Buffer(context, CL_MEM_WRITE_ONLY, pop_size * NB_GENES * NB_GENES * sizeof(double));
					d_fitness = cl::Buffer(context, CL_MEM_WRITE_ONLY, pop_size * sizeof(double));

					// ------------------------------------------------------------------
					// OpenCL initialization of the population
					// ------------------------------------------------------------------

					// Defining gloabl dimmensions (global, size of the whole problem space) and local dimensions (local, size of one workgroup); will stay th esame for both kernels
					cl::NDRange global(pop_size*NB_GENES);
					cl::NDRange local(NB_GENES);

					// Create the compute kernel from the program
					// Don't forget the local and global sizes arguments
					generate_individual(cl::EnqueueArgs(queue, global, local), d_F, d_D, d_A, d_B, d_C, d_permutation, d_X, d_fitness);

					queue.finish();

					cl::copy(queue, d_X, X.begin(), X.end());
					cl::copy(queue, d_fitness, fitness.begin(), fitness.end());

					// ------------------------------------------------------------------
					// OpenCL one generation of the genetic algorithm
					// ------------------------------------------------------------------
					
					cl::Buffer d_operator_probability, d_permutation_parents, d_X_parents, d_fitness_parents;

					// Load in kernel source, creating a program object for the context
					cl::Program program2(context, util::loadProgram("generation.cl"), true);
					// Get the command queue
					cl::CommandQueue queue2(context);

					auto generation_kernel = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(program2, "generation");

					// Stopping criteria:
					// 1. We reach the maximum number of generations OR
					// 2. There have been a certain number of generations we haven't updated the Best solution
					// The main loop
					while( (generation < nb_gen) && (no_improvement < no_improvenment_max) )
					{
						generation++;
						no_improvement++;
						std::cout<<"==============================\n";
						std::cout<<"Generation: "<<generation<<"\n";

						// We create a seed for each workgroup; we hope to be able to update the seeds in the kernels to generate random numbers
						std::vector<unsigned int> operator_probability(pop_size);
						for(int i = 0; i < pop_size; i ++)
							operator_probability[i] = rand();
						d_operator_probability = cl::Buffer(context, operator_probability.begin(), operator_probability.end(), true);

						
						// ------------------------------------------------------------------
						// Setup the buffers and write them into global memory
						// ------------------------------------------------------------------
						// The parents
						d_permutation_parents = cl::Buffer(context, permutation.begin(), permutation.end(), true);
						d_X_parents = cl::Buffer(context, X.begin(), X.end(), true);
						d_fitness_parents = cl::Buffer(context, fitness.begin(), fitness.end(), true);
						// The children
						d_permutation = cl::Buffer(context, CL_MEM_READ_WRITE, pop_size * NB_GENES * sizeof(int));
						d_X = cl::Buffer(context, CL_MEM_WRITE_ONLY, pop_size * NB_GENES * NB_GENES * sizeof(double));
						d_fitness = cl::Buffer(context, CL_MEM_WRITE_ONLY, pop_size * sizeof(double));

						// We'll use the same dimensions and for some, same buffers as in the previous kernel
						generation_kernel(cl::EnqueueArgs(queue2, global, local), d_F, d_D, d_A, d_B, d_C, d_operator_probability, d_permutation_parents, d_X_parents, d_fitness_parents, d_permutation, d_X, d_fitness);
						queue2.finish();

						cl::copy(queue2, d_permutation, permutation.begin(), permutation.end());
						cl::copy(queue2, d_X, X.begin(), X.end());
						cl::copy(queue2, d_fitness, fitness.begin(), fitness.end());
						// We assess how good this generation is; we find if there were improvements and we update the best individual accordingly
						for(int i = 0; i < pop_size; i++){
							// We update the Best solution if we find a better individual
							if(fitness[i] < Best.fitness){
								std::cout<<"---------------- A new Best found\n";
								std::cout<<"Before: "<< Best.fitness;
								//We create a buffer individual that we will copy into Best 
								int buffer_permutation[NB_GENES]; // We copy the permutation associated to the individual in a buffer
								for(int j = 0; j < NB_GENES; j++)
									buffer_permutation[j] = permutation[i*NB_GENES + j];
								Individual I;
								generate_Individual_noRandom(I, buffer_permutation, NB_GENES);
								evaluate_trace(I, F, D);
								copy(Best, I);

								std::cout << " and after: " << fitness[i] << "\n";
								best_generation = generation;
								// We reset the no_improvement iterator
								no_improvement = 0;
							}
						}
					}

				}
				catch (cl::Error err) {
					std::cout << "Exception\n";
					std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
				}

				// We do some tests at the end to know if something went wrong
				printing_test(permutation, X, fitness, 10);
				fitness_test(F, D, permutation, X, fitness, 10);

				t2= clock();
				time= (float)(t2-t1)/CLOCKS_PER_SEC;

				std::cout<<"======================================== Terminated ======================================\n";
				std::cout<<"Best solution found:\n";
				print_permutation(Best);
				std::cout<<"Fitness: " << Best.fitness << "\nExecution time: " << time << " s\nGeneration: " << best_generation << "\n";

				// We check if a solution file has been specified by the user
				if(argc >= 3){
					std::ifstream file_soln;
					file_soln.open(argv[2]);
					if (file_soln.is_open()){
						double Optimal_Value; // The optimal value in the .sln file
						int* optimal_permutation = (int *) calloc(N, sizeof(int));
						int Nsol = open_file_soln(file_soln, Optimal_Value, optimal_permutation);
						if(Nsol != N){
							std::cout<<"The size of the problem in the solution file is different from the data file, you specified the wrong solution file."<<std::endl;
						}
						else{
							std::cout<<"======================================= Known optimal solution ============================\n";
							Individual Optimal;
							Optimal.N = Nsol;
							for(int i = 0; i < Optimal.N; i++)
								Optimal.permutation[i] = optimal_permutation[i];
							construct_matrix(Optimal);

							evaluate_trace(Optimal, F, D);
							std::cout<<"Known optimal solution:\n";
							print_permutation(Optimal);
							std::cout << "Optimal value (in the file): " << Optimal_Value << "\n";
							std::cout << "Optimal value (found by the program): " << Best.fitness << "\n";
						}
						std::free(optimal_permutation);
					}
				}


			}
			else{
				std::cout<<"The dimension (size of a permutaion) provided by the instace file " << argv[1] << " doesn't match the one set in the program (NB_GENES), please check again.\n";
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
