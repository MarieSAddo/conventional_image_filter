/**
 * Parallel implementation of image convolution using OpenMP
 * to compile this file, run the following:
 * gcc-13 -o convolve_parallel_cpu convolve_parallel_cpu.c -lm -lpng -fopenmp
 * mpirun -np 4 ./convolve_parallel_cpu 128 256 3
 */
#include <stdio.h>
#include <stdlib.h>
#include "kernels.h"
#include "image.h"
#include <time.h>
#include <omp.h>
#include <string.h>

// #define KERNEL_SIZE 3

int clamp(double value, int min, int max)
{
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return (int)value;
}
void convolve(Image *img, double **kernel, int kernel_size, Image *output_img)
{
    //int kernel_size_half = kernel_size / 2;
// Perform convolution
#pragma omp parallel for default(none) shared(img, kernel, kernel_size, output_img)
    for (int i = 0; i < img->height; i++) // Iterate over the rows of the image
    {
        for (int j = 0; j < img->width; j++)
        {
            // Initialize the output pixel value
            double output_pixel = 0;
            // Iterate over the kernel
            for (int k = 0; k < kernel_size; k++) // Iterate over the rows of the kernel
            {
                for (int l = 0; l < kernel_size; l++) // Iterate over the columns of the kernel
                {
                    // Calculate the coordinates of the pixel in the input image
                    int x_index = i + k - kernel_size /2;
                    int y_index = j + l - kernel_size /2;
                    // Check if the pixel is within the bounds of the image
                    if (x_index >= 0 && x_index < img->height && y_index >= 0 && y_index < img->width)
                    {
                        // Multiply the kernel value with the corresponding pixel value in the input image
                        output_pixel += kernel[k][l] * (double)img->data[x_index][y_index];
                    }
                }
            }
            // Set the output pixel value in the output image
            // Ensure the output_pixel value is within the range of pixel values
            output_img->data[i][j] = clamp(output_pixel, 0, 255); // You need to implement clamp function;

        }
    }
}

void free_kernel(double **kernel)
{
    free(kernel);
}

int main()
{
    // omp_set_num_threads(4);

    srand(0); // Seed the random number generator with the current time to get different random numbers each time the program is run
    int kernel_size = (rand() % 5) * 2 + 3; // Randomly generate a kernel size between 3 and 11 that is an odd number
    
    // Read the PNG file
    Image img;
    read_png_file("image1.png", PNG_COLOR_TYPE_GRAY, &img);

    // Prompt user to enter a kernel type they want to use
    char kernel_name[50];
    printf("Enter the type of kernel you want to use (gauss, unsharpen_mask, mean): ");
    scanf("%s", kernel_name);

    // Generate kernel based on the user input
    double **kernel;

    if (strcmp(kernel_name, "gauss") == 0)
    {
        kernel = gauss_kernel(kernel_size);
    }
    else if (strcmp(kernel_name, "unsharpen_mask") == 0)
    {
        kernel = unsharpen_mask_kernel(kernel_size);
    }
    else if (strcmp(kernel_name, "mean") == 0)
    {
        kernel = mean_kernel(kernel_size);
    }
    else
    {
        printf("Invalid kernel type\n");
        return 1;
    }

    // Allocate memory for the output image
    Image output_img;
    output_img.width = img.width;
    output_img.height = img.height;
    output_img.color_type = PNG_COLOR_TYPE_GRAY;
    malloc_image_data(&output_img);

    // check if the data field of img and output_img are not null
    if (img.data == NULL || output_img.data == NULL)
    {
        printf("Error: Memory not allocated for image data\n");
        return 1;
    }

    // start the clock
    clock_t start_time = clock();

    // Perform convolution
    convolve(&img, kernel, kernel_size, &output_img);

    // get the end time and calculate the elapsed time
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    FILE *f = fopen("parallel_cpu_time.md", "a");
    if (f != NULL)
    {
        fprintf(f, "Kernel: %s, Time: %f, Size: %d\n", kernel_name, elapsed_time, kernel_size);
        fclose(f);
    }
    else
    {
        printf("Error opening file!\n");
    }

    // write to an output file based on the kernel type
    char output_file[128];
    snprintf(output_file, sizeof(output_file), "serial_output_%s.png", kernel_name);

    // Free the memory allocated for the images
    free_image_data(&img);
    free_image_data(&output_img);
    free_kernel(kernel);
    return 0;
}
