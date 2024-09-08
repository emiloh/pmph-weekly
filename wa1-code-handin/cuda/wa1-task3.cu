#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer.h"

#define GPU_RUNS 300
#define MAX_N 1000000000

__global__ void Kernel(float *in, float *out, unsigned int N) {
  // Find the global id
  const unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;

  if (gid < N) {
    const float temp = in[gid] / (in[gid] - 2.3);
    out[gid] = temp * temp * temp;
  }
}

void run_cpu(unsigned int N, float *in, float *out, struct timeval *t_start);
void run_gpu(unsigned int N, float *d_in, float *d_out,
             struct timeval *t_start, unsigned int gpu_runs);
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
  float *h_in = (float *)malloc(mem_size);
  float *h_out_cpu = (float *)malloc(mem_size);
  float *h_out_gpu = (float *)malloc(mem_size);

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

  cudaMalloc((void **)&d_in, mem_size);
  cudaMalloc((void **)&d_out, mem_size);

  // Copy memory from host to device
  cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

  run_gpu(N, d_in, d_out, &t_start, GPU_RUNS);

  // Check for errors
  cudaError_t gpu_code = cudaPeekAtLastError();
  if (gpu_code != cudaSuccess) {
    printf("GPU error: %s\n", cudaGetErrorString(gpu_code));
    exit(1);
  }

  cudaMemcpy(h_out_gpu, d_out, mem_size, cudaMemcpyDeviceToHost);

  validate(N, h_out_cpu, h_out_gpu);

  // Free memory
  free(h_in);
  free(h_out_cpu);
  free(h_out_gpu);
  cudaFree(d_in);
  cudaFree(d_out);
}

void run_cpu(unsigned int N, float *in, float *out, struct timeval *t_start) {
  timer_start(t_start);

  for (unsigned int i = 0; i < N; ++i) {
    out[i] = powf((in[i] / (in[i] - 2.3)), 3.0);
  }
  
  timer_end_cpu(t_start);
}

void run_gpu(unsigned int N, float *in, float *out, struct timeval *t_start,
             unsigned int gpu_runs) {
  timer_start(t_start);

  for (unsigned int i = 0; i < gpu_runs; ++i) {
    // Setup grid and blocks
    unsigned int blocks = 512;
    unsigned int num_blocks = (N + blocks - 1) / blocks;
    dim3 block(blocks, 1, 1), grid(num_blocks, 1, 1);

    Kernel<<<grid, block>>>(in, out, N);
  }

  cudaDeviceSynchronize();

  timer_end_gpu(t_start, N, GPU_RUNS);
}

void validate(unsigned int N, float *cpu_out, float *gpu_out) {
  float epsilon = 1.0e-5;

  for (unsigned int i = 0; i < N; ++i) {
    if (fabs(gpu_out[i] - cpu_out[i]) > epsilon) {
      printf("Unsuccessful validation - gpu: %f, cpu: %f\n", gpu_out[i],
             cpu_out[i]);
      exit(1);
    }
  }

  printf("Successful validation!\n");
}
