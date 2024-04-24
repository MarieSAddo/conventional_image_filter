#!/bin /bash

#Compile the C program
nvcc -arch=sm_86 -O3 --compiler-options -march=native convole_parallel_gpu.cu -o my_program -lm

# Run the C program
./my_program 

