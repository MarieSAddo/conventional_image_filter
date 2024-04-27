#!/bin/bash 
#SBATCH -p shared
#SBATCH --nodes 2           
#SBATCH --tasks-per-node=2
#SBATCH -t 01:00:00
#SBATCH -A mor101
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=ALL

module load intelmpi
module load openmpi/4.1.6

gcc -Wall -O3 $(libpng-config --I_opts) -fopenmp image.c kernels.c convolve_parallel_openmp.c -o openmp_program $(libpng-config --L_opts) -lpng

# Run the C program
srun -n 4 –cpus-per-task 2 ./openmp_program 


# // void write_results_to_file(const char *filename, const char *results) {
# //     FILE *file = fopen(filename, "a");
# //     if (file == NULL) {
# //         printf("Error opening file!\n");
# //         return;
# //     }

# //     // Get the current time
# //     time_t t = time(NULL);
# //     struct tm *tm = localtime(&t);
# //     char time_str[64];
# //     strftime(time_str, sizeof(time_str), "%c", tm);

# //     // Write the time and results to the file
# //     fprintf(file, "## %s\n\n", time_str);
# //     fprintf(file, "%s\n", results);

# //     fclose(file);
# // }

    # // // print the number of threads
    #         // printf("Number of threads: %d\n", omp_get_num_threads());
    #         // //print the number of rows
    #         // printf("Number of rows: %d\n", img->height);
    #         // //print the number of columns
    #         // printf("Number of columns: %d\n", img->width);
    #         // //print the kernel size
    #         // printf("Kernel size: %d\n", kernel_size);