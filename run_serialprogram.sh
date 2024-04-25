#!/bin /bash

#Compile the C program
gcc -Wall -O3 $(libpng-config --I_opts) image.c kernels.c convolve_serial_cpu.c -o my_program $(libpng-config --L_opts) -lpng

# Run the C program
./my_program 

