#!/bin /bash

#Compile the C program
nvcc -o convolve_parallel_gpu convolve_parallel_gpu.cu -lm -arch=sm_86 --compiler-options -march=native -O3 

#nvcc -arch=sm_86 -O3 --compiler-options -march=native convole_parallel_gpu.cu -o my_gpu_program -lm

# Run the C program
./convole_parallel_gpu

