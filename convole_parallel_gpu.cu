// to compile this file use the following command in the terminal "nvcc convole_parallel_gpu.cu -lpng -o convole_parallel_gpu"
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
#include "kernels.c"

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

// #define KERNEL_SIZE 3

// CUDA kernel for convolution which processes
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

// Launches multiple convolution kernels for each image and kernel combination
void parallel_convolve(Image *images, int num_images, double **kernels, int num_kernels, Image *output_images)
{
    double *d_kernels;
    int total_kernel_size = num_kernels * kernel_sizes * kernel_sizes;
    cudaMalloc((void **)&d_kernels, total_kernel_size * sizeof(double));
    cudaMemcpy(d_kernels, kernels[0], total_kernel_size * sizeof(double), cudaMemcpyHostToDevice);
    for (int i = 0; i < num_images; i++)
    {
        unsigned char *d_image;
        cudaMalloc((void **)&d_image, images[i].width * images[i].height * sizeof(unsigned char));
        cudaMemcpy(d_image, images[i].data, images[i].width * images[i].height * sizeof(unsigned char), cudaMemcpyHostToDevice);
        unsigned char *d_output_image;
        cudaMalloc((void **)&d_output_image, images[i].width * images[i].height * sizeof(unsigned char));
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((images[i].width + threadsPerBlock.x - 1) / threadsPerBlock.x, (images[i].height + threadsPerBlock.y - 1) / threadsPerBlock.y);
        convolve<<<numBlocks, threadsPerBlock>>>((unsigned char *)images[i].data, d_kernels, kernel_sizes, images[i].width, images[i].height, (unsigned char *)output_images[i].data);
        cudaMemcpy(output_images[i].data, d_output_image, images[i].width * images[i].height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        {
            cudaFree(d_image);
            cudaFree(d_output_image);
        }
    }
    cudaFree(d_kernels);
}

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
            for (int i = 0; i < num_sizes; i++)
            {
                int kernel_sizes = kernel_sizes[i];
                for (int j = 0; j < 3; j++)
                {
                    double **kernel = NULL;
                    if (strcmp(kernel_names[i], "gauss") == 0)
                    {
                        kernel = gauss_kernel(kernel_sizes[i]);
                    }
                    else if (strcmp(kernel_names[i], "unsharpen_mask") == 0)
                    {
                        kernel = unsharpen_mask_kernel(kernel_sizes[i]);
                    }
                    else if (strcmp(kernel_names[i], "mean") == 0)
                    {
                        kernel = mean_kernel(kernel_sizes[i]);
                    }
                    // Allocate memory for the output image
                    Image output_img;
                    output_img.width = img.width;
                    output_img.height = img.height;
                    output_img.color_type = PNG_COLOR_TYPE_GRAY;
                    malloc_image_data(&output_img);
                    // Allocate memory for the kernel on the GPU
                    double *d_kernel;
                    cudaMalloc(&d_kernel, kernel_sizes * kernel_sizes * sizeof(double));
                    cudaMemcpy(d_kernel, kernel, kernel_sizes * kernel_sizes * sizeof(double), cudaMemcpyHostToDevice);
                    // Perform convolution
                    clock_t start_time = clock();
                    dim3 threadsPerBlock(16, 16); // You can adjust these values as needed
                    dim3 numBlocks((img.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (img.height + threadsPerBlock.y - 1) / threadsPerBlock.y);
                    convolve<<<numBlocks, threadsPerBlock>>>((unsigned char *)img.data, d_kernel, kernel_sizes, img.width, img.height, (unsigned char *)output_img.data);
                    // convolve((unsigned char *)img.data[0], d_kernel, KERNEL_SIZE, img.width, img.height, (unsigned char *)output_img.data[0]);
                    clock_t end_time = clock();
                    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
                    // Write results to the markdown file
                    // Write results to the markdown file
                    FILE *f = fopen("parallel_gpu_time.md", "a");
                    if (f != NULL)
                    {
                        int result = fprintf(f, "%s, %s, %f, %d\n", entry->d_name, kernel_names[i], elapsed_time, kernel_sizes);
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
