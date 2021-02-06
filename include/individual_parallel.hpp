#ifndef _INDIVIDUAL_HPP_
#define _INDIVIDUAL_HPP_

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


#endif
