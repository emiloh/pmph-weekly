#ifndef TIMER
#define TIMER

#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

#define MICROSEC_PER_SEC 1000000

void timer_start(struct timeval *t_start)
{
	gettimeofday(t_start, NULL);
}

void timer_end_cpu(struct timeval *t_start)
{
	struct timeval t_end;
	gettimeofday(&t_end, NULL);
	
	long int diff = (t_end.tv_usec - t_start->tv_usec) + MICROSEC_PER_SEC * (t_end.tv_sec - t_start->tv_sec);

	printf("CPU elapsed time (microseconds): %ld\n", diff);

}

void timer_end_gpu(struct timeval *t_start, unsigned int N, unsigned int GPU_RUNS)
{
	struct timeval t_end;
	gettimeofday(&t_end, NULL);
	
	long int diff  = (t_end.tv_usec - t_start->tv_usec) + MICROSEC_PER_SEC * (t_end.tv_sec - t_start->tv_sec);
	double elapsed_microsecs = diff/GPU_RUNS;

	// numerator is multiplied with 2, to account for 1 read and 1 write
	double GBpS    = (2.0 * N * 4.0) / (elapsed_microsec * 1000)
	
	printf("Kernel average elapsed time (microseconds): %f. Throughput (GB/sec): %f \n", elapsed, GBpS);
}

#endif
