#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer.h"

#define GPU_RUNS 300
#define MAX_N 500000000

__global__ void Kernel(float *x) {
  // fill me
}

void run_cpu(unsigned int N, float *in, float *out, struct timeval *t_start);
void run_gpu(unsigned int N, float *h_in, float *h_out, float *d_in,
             float *d_out, struct timeval *t_start);
void validate(unsigned int N, float *cpu_out, float *gpu_out);

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Invalid number of arguments: %d. Expected 1!\n", argc);
    exit(1);
  }

  unsigned int N = atoi(argv[1]);
  printf("Size of array: %d\n", N);

  if (N > MAX_N) {
    printf("Size of array too big. Maximal size is %d.\n", MAX_N);
    exit(1);
  }
  
  unsigned int mem_size = N * sizeof(float);

  // Allocating host memory
  float *h_in      = (float*) malloc(mem_size);
  float *h_out_cpu = (float*) malloc(mem_size);
  float *h_out_gpu = (float*) malloc(mem_size);

  for (unsigned int i = 0; i < N; ++i) {
    h_in[i] = (float)i;
  }

  // Timer
  struct timeval t_start;

  // Run sequentially
  run_cpu(N, h_in, h_out_cpu, &t_start);

  // Allocate device memory
  float *d_in;
  float *d_out;

  // Free memory
  free(h_in);
  free(h_out_cpu);
  free(h_out_gpu);
  free(d_in);
  free(d_out);
}

void run_cpu(unsigned int N, float *in, float *out, struct timeval *t_start) {
  timer_start(t_start);

  for (unsigned int i = 0; i < N; ++i) {
    out[i] = powf((in[i] / (in[i] - 2.3)), 3.0);
  }
  printf("Done, last element: %f\n", out[N - 1]);
  timer_end_cpu(t_start);
}

