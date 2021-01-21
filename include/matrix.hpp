#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <vector>
#include <array>
#include <algorithm>
#include <random>

#include <iostream>
#include <fstream>


// Compute the transpose of a matrix
void mat_transpose(int, std::vector<double>&, std::vector<double>&);

// Compute the trace of a matrix
double mat_trace(int, std::vector<double>&);

// Compute the product of two matrices A and B
void seq_mat_mul_sdot(int, std::vector<double>&, std::vector<double>&, std::vector<double>&);

// Print a matrix
// n is the number of rows and m of columns
void print_matrix(std::vector<double>, int, int);


#endif
