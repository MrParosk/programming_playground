output: main.o matrix.o utils.o serial_gradient_descent.o omp_gradient_descent.o
	gcc main.o matrix.o utils.o serial_gradient_descent.o omp_gradient_descent.o -fopenmp -o output

main.o: main.c
	gcc -c main.c

matrix.o: src/matrix.c src/matrix.h
	gcc -c src/matrix.c

serial_gradient_descent.o: src/serial_gradient_descent.c src/serial_gradient_descent.h
	gcc -c src/serial_gradient_descent.c

omp_gradient_descent.o: src/omp_gradient_descent.c src/omp_gradient_descent.h
	gcc -fopenmp -c src/omp_gradient_descent.c

utils.o: src/utils.c src/utils.h
	gcc -c src/utils.c

clean:
	del *.o
	del output.exe