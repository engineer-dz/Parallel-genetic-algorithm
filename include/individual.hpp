#ifndef _INDIVIDUAL_HPP_
#define _INDIVIDUAL_HPP_

#include <vector>
#include <array>
#include <algorithm>
#include <random>

#include <iostream>
#include <fstream>


class Individual
{
private:
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


#endif
