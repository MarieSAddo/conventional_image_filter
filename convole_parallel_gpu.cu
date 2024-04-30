/*
* to compile this file use the following command in the terminal "nvcc convole_parallel_gpu.cu -lpng -o convole_parallel_gpu
* make sure youre in a shared node with a GPU to run this file ./convole_parallel_gpu
*/
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h> // For directory handling
#include <string.h>
#include <png.h>
#include "image.h"
#include <time.h>
#include <cuda_runtime.h>
#include "kernels.h"
#include "image.h"


// Macro check if CUDA call has an error, and if it does report it and exit the program
#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        const cudaError_t error = (call);                                      \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    } while (0)



// CUDA kernel for convolution which processes
/**
 * This function peforms a CUDA kernel for convolution which processes the image in parallel on the GPU.
 * It takes a 2D image and a 2D kernel as input and applies the convolution operation to the image using the kernel.
 * The convolution operation is applied to each pixel in the image in parallel by each thread in the GPU.
 * @param d_image: The input image data 
 * @param d_kernel: The kernel data 
 * @param kernelSize: The size of the kernel
 * @param width:The width of the image
 * @param height: The height of the image
 * @param d_output_image: The output image data
 * 
*/
__global__ void convolve(unsigned char *d_image, double *d_kernel, int kernelSize, int width, int height, unsigned char *d_output_image)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of the pixel to process (x-coordinate)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of the pixel to process (y-coordinate)
    if (col < width && row < height)
    {
        double sum = 0;
        int kernelCenter = kernelSize / 2; // Calculate the center of the kernel
        for (int i = -kernelCenter; i <= kernelCenter; i++)
        {
            for (int j = -kernelCenter; j <= kernelCenter; j++)
            {
                int curRow = row + i;
                int curCol = col + j; // calculate the current row and column index of the kernel element being processed
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                {
                    sum += d_image[curRow * width + curCol] * d_kernel[(i + kernelCenter) * kernelSize + (j + kernelCenter)];
                }
            }
        }
        d_output_image[row * width + col] = (unsigned char)sum; // Store the result in the output image
    }
}


/*
* The main function reads the images from the images directory. It has a list of kernel names and sizes. 
* It loops through each image and each kernel type and size, references the kernel function, allocates memory for the output image, 
* performs convolution on the image using the kernel, and writes the results to a markdown file.
* The function returns 0 if the program runs successfully, otherwise it returns 1
* The results in the markdown file are from being run on the GPU
* Included print statements to indicate if the main was sucessful running.
*/
int main()
{
    srand(0); // Seed the random number generator with the current time to get different random numbers each time the program is run
    DIR *dir = opendir("images");
    if (dir == NULL)
    {
        printf("Error: Unable to open the images directory\n");
        return 1;
    } 
    // Array to store kernel names (assuming a limited number of kernels)
    char kernel_names[3][50] = {"gauss", "unsharpen_mask", "mean"};

    int kernel_sizes[] = {3, 9, 15, 25, 49};
    int num_sizes = sizeof(kernel_sizes) / sizeof(kernel_sizes[0]);
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        {
            continue; // Skip "." and ".." entries in the directory
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
            for (int i = 0; i < num_sizes; i++)
            {
                int kernel_size = kernel_sizes[i];
                for (int j = 0; j < 3; j++)
                {
                    double **kernel = NULL;
                    if (strcmp(kernel_names[i], "gauss") == 0)
                    {
                        kernel = gauss_kernel(kernel_size);
                    }
                    else if (strcmp(kernel_names[i], "unsharpen_mask") == 0)
                    {
                        kernel = unsharpen_mask_kernel(kernel_size);
                    }
                    else if (strcmp(kernel_names[i], "mean") == 0)
                    {
                        kernel = mean_kernel(kernel_size);
                    }
                    // Allocate memory for the output image
                    Image output_img;
                    output_img.width = img.width;
                    output_img.height = img.height;
                    output_img.color_type = PNG_COLOR_TYPE_GRAY;
                    malloc_image_data(&output_img);
                    // Allocate memory for the kernel on the GPU
                    double *d_kernel;
                    cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(double));
                    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(double), cudaMemcpyHostToDevice);
                    // Perform convolution
                    clock_t start_time = clock();
                    dim3 threadsPerBlock(16, 16); // You can adjust these values as needed
                    dim3 numBlocks((img.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (img.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
                    convolve<<<numBlocks, threadsPerBlock>>>((unsigned char *)img.data, d_kernel, kernel_size, img.width, img.height, (unsigned char *)output_img.data);
                    // convolve((unsigned char *)img.data[0], d_kernel, KERNEL_SIZE, img.width, img.height, (unsigned char *)output_img.data[0]);
                    clock_t end_time = clock();
                    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
                    
                    // Write results to the markdown file
                    FILE *f = fopen("parallel_gpu_time.md", "a");
                    if (f != NULL)
                    {
                        int result = fprintf(f, "%s, %s, %f, %d\n", entry->d_name, kernel_names[i], elapsed_time, kernel_size);
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
                    cudaFree(kernel);
                }
            }
        }
    }
    closedir(dir);
    return 0;
}
