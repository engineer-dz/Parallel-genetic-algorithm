#include <iostream>
#include <cstdint>
#include <climits>
#define N 26

uint xorshift32(uint seed){
	uint x = seed;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}

float rand_float(uint random){
	return random/(float)UINT_MAX;
}

int rand_int(uint random, int n){
	return random%n;
}

/*
int main(int argc, char *argv[]){
	srand(time(NULL));
	uint seed = rand();
	std::cout<<"returned random   ";
	std::cout<<"[0, 1]   ";
	std::cout<<"binary    ";
	std::cout<<"int    ";
	std::cout<<std::endl;
	for(int i = 0; i < 1000; i++){
		seed = xorshift32(seed);
		std::cout<<seed<<" ";
		std::cout<<rand_float(seed)<<" ";
		std::cout<<rand_int(seed, 2)<<" ";
		std::cout<<rand_int(seed, N)<<" ";
		std::cout<<std::endl;
	}
}
*/