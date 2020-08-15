test: *.c
	g++ -fopenmp -w -c Winograd_convolution.c -lm
	g++ -fopenmp -w Winograd_convolution.o -lm -o test
	rm -f *.o
clean:
	rm -f test *.o