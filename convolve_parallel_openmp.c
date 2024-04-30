/**
 * to compile: gcc -o convolve_parallel_openmp convolve_parallel_openmp.c image.c kernels.c -lm -lpng 
 * srun -n 1 --cpus-per-task=2 ./convolve_parallel_openmp
 */
#include <stdio.h>
#include <dirent.h>
#include <png.h>
#include <stdlib.h>
#include "kernels.h"
#include "image.h"
#include <time.h>
#include <omp.h>
#include <string.h>

/**
 * This function clamps a double value to the range [min, max] to ensure it is within the range of pixel values.
 * @param value The value to clamp
 * @param min The minimum value
 * @param max The maximum value
 * @return The clamped value
*/
int clamp(double value, int min, int max)
{
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return (int)value;
}

/**
 * This function performs convolution on an image using a given kernel.
 * It takes an input image, a kernel, kernelsize and an output image as arguments.
 * The function convolves the input image with the kernel and stores the result in the output image.
 * #pragma omp parallel for default(none) shared(img, kernel, kernel_size, output_img) 
 * @param img The input image
 * @param kernel The kernel to use for convolution
 * @param kernel_size The size of the kernel
 * @param output_img The output image
 * @return void
*/
void convolve(Image *img, double **kernel, int kernel_size, Image *output_img)
{
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
                    int x_index = i + k - kernel_size / 2;
                    int y_index = j + l - kernel_size / 2;
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


/**
* The main function reads the images from the images directory. It has a list of kernel names and sizes. 
* It loops through each image and each kernel type and size, references the kernel function, allocates memory for the output image, 
* performs convolution on the image using the kernel, and writes the results to a markdown file.
* The function returns 0 if the program runs successfully, otherwise it returns 1
* @return 0 if the program runs successfully, otherwise 1
* Included print statements to show the progress of the program
*/
int main()
{
    
    printf("Starting convolution...\n");

    srand(0); // Seed the random number generator with the current time to get different random numbers each time the program is run
    DIR *dir = opendir("images");
    if (dir == NULL)
    {
        printf("Error: Unable to open the images directory\n");
        return 1;
    }

    // Array to store kernel names (assuming a limited number of kernels)
    char kernel_names[3][50] = {"gauss", "unsharpen_mask", "mean"};
    int kernel_sizes[]= {3, 9, 15, 25, 49};
    printf("Reading images...\n");
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
         printf("Processing file: %s\n", entry->d_name);

        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        {
            continue; // Skip "." and ".." entries
        }

        // Check if the entry is a regular file (image)
        if (entry->d_type & DT_REG)
        {
            char image_path[128];
            snprintf(image_path, sizeof(image_path), "images/%s", entry->d_name);

            // Read the image
            Image img;
            read_png_file(image_path, PNG_COLOR_TYPE_GRAY, &img);

            // Loop through each kernel type
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < sizeof(kernel_sizes)/sizeof(kernel_sizes[0]); j++) {
                double **kernel = NULL;
                if (strcmp(kernel_names[i], "gauss") == 0)
                {
                    kernel = gauss_kernel(kernel_sizes[j]);
                }
                else if (strcmp(kernel_names[i], "unsharpen_mask") == 0)
                {
                    kernel = unsharpen_mask_kernel(kernel_sizes[j]);
                }
                else if (strcmp(kernel_names[i], "mean") == 0)
                {
                    kernel = mean_kernel(kernel_sizes[j]);
                }
                printf("Processing file: %s with kernel: %s\n", entry->d_name, kernel_names[i]);

                // Allocate memory for the output image
                Image output_img;
                output_img.width = img.width;
                output_img.height = img.height;
                output_img.color_type = PNG_COLOR_TYPE_GRAY;
                malloc_image_data(&output_img);

                // Perform convolution
                clock_t start_time = clock();
                convolve(&img, kernel, kernel_sizes[j], &output_img);
                clock_t end_time = clock();
                double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

                // Write results to the markdown file
                FILE *f = fopen("parallel_cpu_time.md", "a");
                if (f != NULL)
                {
                    int result = fprintf(f, "%s, %s, %f, %d\n", entry->d_name, kernel_names[i], elapsed_time, kernel_sizes[j]);
                    if (result < 0)
                    {
                        perror("Error writing to file");
                    }
                    else
                    {
                        printf("Successfully wrote to file.\n");
                    }
                    fclose(f);
                }
                else
                {
                    perror("Error opening file");
                }
                free_image_data(&output_img);
                free_kernel(kernel);
                }
            }
        }
    }

    closedir(dir);

    return 0;
}
