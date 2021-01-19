name = qap_ga
objects =  $(name).o 
header = 
program = $(name).out

install: $(objects)
	g++ -O2 $(objects) -o $(program)

$(name).o : $(header) 

clean:
	@rm -f $(program) $(objects) .[!.]* ../.[!.]*

clear:
	@rm -f $(objects) .[!.]* ../.[!.]*

