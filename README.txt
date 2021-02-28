Commands to execute the program without comparison to the known solution:

	mkdir build
	cd build
	cmake ..
	make -j8
	cd src_parallel
	./qap_pga ../data/bur26a.dat


Commands to execute the program and compare the found solution to the known solution:

	mkdir build
	cd build
	cmake ..
	make -j8
	cd src_parallel
	./qap_pga ../data/bur26a.dat ../data/bur26a.sln


If the size of the problem is changed, that is, if the number of genes is different, the preprocessor constant NB_GENES must be changed in the code, in file include/individual_parallel.hpp and the sources must be rebuilt.
