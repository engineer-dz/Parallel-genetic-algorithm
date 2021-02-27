#ifndef _INDIVIDUAL_HPP_
#define _INDIVIDUAL_HPP_
// Fixed parameters: size of the population, the maximum number of generations (first stopping criterion),
// the maximum number of generations where we didn't improve the Best solution (second stopping criterion)
// and size of an individual (length of the permutation matrix; number of genes)

#define pop_size 1000
#define nb_gen 250
#define no_improvenment_max 25
#define NB_GENES 26

#include <vector>
#include <array>
#include <algorithm>
#include <random>

#include <iostream>
#include <fstream>


typedef struct Individual
{
	int N = NB_GENES; // size of the permutation vector
	int permutation[NB_GENES]; // permutation vector
	double X[NB_GENES * NB_GENES]; // permutation matrix representation
	double fitness; // value of the objective function for this solution
	
} Individual;


void generate_Individual(Individual &I, int n);
void generate_Individual_noRandom(Individual &I, int *p, int n);
void construct_matrix(Individual &I);
void evaluate_trace(Individual &I, const std::vector<double> &F, const std::vector<double> &D);
double evaluate_original(int* p, const int &n, const std::vector<double> &F, const std::vector<double> &D);
void mutate(Individual &I);
void crossover(const Individual& Individual1, const Individual& Individual2, Individual &Offspring);
void heuristic_2opt(Individual &I, const std::vector<double> &D, const std::vector<double> &F);
void print_permutation(const Individual &I);
void print_matrix(const Individual &I);
void copy(Individual &Dest, const Individual &Source);
void printing_test(std::vector<int> permutation, std::vector<double> X, std::vector<double> fitness, int sample_size);
void fitness_test(std::vector<double> F, std::vector<double> D, std::vector<int> permutation, std::vector<double> X, std::vector<double> fitness, int sample_size);
int open_file_dat(std::ifstream &file_dat, std::vector<double> &Flow, std::vector<double> &Distance);
int open_file_soln(std::ifstream &file_soln, double &Value, int *Solution);

#endif
