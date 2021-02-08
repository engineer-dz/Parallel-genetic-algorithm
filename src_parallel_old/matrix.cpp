
#include <vector>
#include <array>
#include <algorithm>
#include <random>

#include <iostream>
#include <fstream>

#include "matrix.hpp"


// Compute the transpose of a matrix
void mat_transpose(int N, const std::vector<double> &A, std::vector<double> &At)
{
	// We empty the vector
	At.clear();
	At.reserve(N*N);

	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			At[i*N + j] = A[j*N + i];
}


// Compute the trace of a matrix
double mat_trace(int N, const std::vector<double> &A)
{
	double tmp = 0;
	for(int i = 0; i < N; i++)
		tmp += A[i*N + i];
	return tmp;
}


// Compute the product of two matrices A and B
void seq_mat_mul_sdot(int N, const std::vector<double> &A, const std::vector<double> &B, std::vector<double> &C)
{
	double tmp;
	// We empty the vector
	C.clear();
	C.reserve(N*N);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			tmp = 0.0f;
			for (int k = 0; k < N; k++) {
				tmp += A[i*N+k] * B[k*N+j];
			}
			C[i*N+j] = tmp;
		}
	}
}


// Print a matrix
// n is the number of rows and m of columns
void print_matrix(const std::vector<double> &matrix, int n, int m)
{
	for(int i = 0; i < n; i++){
		std::cout << std::endl;
		for(int j = 0; j < n; j++)
			std::cout << matrix[n*i + j] << " ";
	}
	std::cout << std::endl;
}
