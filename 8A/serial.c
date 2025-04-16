#include <stdio.h>
#include <math.h>
#include <time.h>

int main ( int argc, char *argv[] ){
	const long int N = 10000000000;
	const double h = 1.0/N;
	const double PI = 3.141592653589793238462643;
	double x, sum, pi, error, time; int i;

	clock_t start_time, end_time;

	start_time = clock();

	sum = 0.0;
	for ( i = 0; i <= N; i++ ){
		x = h * (double)i;
		sum += 4.0/( 1.0 + x * x );
	}

	pi = h * sum;
	
	end_time = clock();

	time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

	error = pi - PI;
	error = error < 0 ? -error:error;
	printf("pi = %18.16f +/- %18.16f\n", pi, error);
	printf("time = %18.16f sec\n", time);

	return 0;
}
