COMPILER   = ncc
FLAGS      = -O3
SOURCE     = wa1-task3.cu 
EXECUTABLE = wa1-task3
DEFAULT    = 753411


default: 
	clean compile run

clean: 
	rm $(EXECUTABLE)

compile: 
	$(COMPILER) $(FLAGS) -o $(EXECUTABLE) $(SOURCE) 

run: 
	./$(EXECUTABLE)

run_specific:
	./$(EXECUTABLE) $(N) 
  
