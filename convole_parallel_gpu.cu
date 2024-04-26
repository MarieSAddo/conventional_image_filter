#include <stdio.h>
#include <stdlib.h>
#include <dirent.h> // For directory handling
#include <string.h>
#include <png.h>
#include "image.h"
#include <time.h>
#include <cuda_runtime.h>

#define KERNEL_SIZE 3

// CUDA kernel for convolution
__global__ void convolutionKernel(const unsigned char* img, const double* kernel, int imgWidth, int imgHeight, unsigned char* output) {
    // Calculate the index of the thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < imgWidth && idy < imgHeight) {
        // Initialize the output pixel value
        double output_pixel = 0;
        // Calculate the coordinates of the pixel in the input image
        int x_index = idy - KERNEL_SIZE / 2;
        int y_index = idx - KERNEL_SIZE / 2;
        // Iterate over the kernel
        for (int k = 0; k < KERNEL_SIZE; k++) // Iterate over the rows of the kernel
        {
            for (int l = 0; l < KERNEL_SIZE; l++) // Iterate over the columns of the kernel
            {
                // Check if the pixel is within the bounds of the image
                int x_temp = x_index + k;
                int y_temp = y_index + l;
                if (x_temp >= 0 && x_temp < imgHeight && y_temp >= 0 && y_temp < imgWidth)
                {
                    // Multiply the kernel value with the corresponding pixel value in the input image
                    output_pixel += kernel[k * KERNEL_SIZE + l] * (double)img[x_temp * imgWidth + y_temp];
                }
            }
        }
        // Set the output pixel value in the output image
        // Ensure the output_pixel value is within the range of pixel values
        output[idy * imgWidth + idx] = (unsigned char)clamp(output_pixel, 0, 255);
    }
}

void convolve(Image *img, double *kernel, int kernel_size, Image *output_img) {
    // Define CUDA memory pointers
    unsigned char *d_img, *d_output;
    double *d_kernel;

    // Allocate GPU memory
    cudaMalloc((void**)&d_img, img->width * img->height * sizeof(unsigned char));
    cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * sizeof(double));
    cudaMalloc((void**)&d_output, output_img->width * output_img->height * sizeof(unsigned char));

    // Transfer data from CPU to GPU
    cudaMemcpy(d_img, img->data[0], img->width * img->height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((img->width + blockSize.x - 1) / blockSize.x, (img->height + blockSize.y - 1) / blockSize.y);

    // Launch CUDA kernel
    convolutionKernel<<<gridSize, blockSize>>>(d_img, d_kernel, img->width, img->height, d_output);

    // Transfer results back from GPU to CPU
    cudaMemcpy(output_img->data[0], d_output, output_img->width * output_img->height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_img);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main() {
    srand(0);

    // Open the "images" directory
    DIR *dir = opendir("images");
    if (dir == NULL) {
        printf("Error opening images directory\n");
        return 1;
    }

    // Array to store kernel names (assuming a limited number of kernels)
    char kernel_names[3][50] = {"gauss", "unsharpen_mask", "mean"};

    // Loop through all files in the directory
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue; // Skip "." and ".." entries
        }

        // Check if the entry is a regular file (image)
        if (entry->d_type & DT_REG) {
            char image_path[128];
            sprintf(image_path, "images/%s", entry->d_name);

            // Read the image
            Image img;
            read_png_file(image_path, PNG_COLOR_TYPE_GRAY, &img);

            // Loop through each kernel type
            for (int i = 0; i < 3; i++) {
                double *kernel = NULL;
                if (strcmp(kernel_names[i], "gauss") == 0) {
                    kernel = gauss_kernel(KERNEL_SIZE);
                } else if (strcmp(kernel_names[i], "unsharpen_mask") == 0) {
                    kernel = unsharpen_mask_kernel(KERNEL_SIZE);
                } else if (strcmp(kernel_names[i], "mean") == 0) {
                    kernel = mean_kernel(KERNEL_SIZE);
                }

                // Allocate memory for the output image
                Image output_img;
                output_img.width = img.width;
                output_img.height = img.height;
                output_img.color_type = PNG_COLOR_TYPE_GRAY;
                malloc_image_data(&output_img);

                // Perform convolution
                clock_t start_time = clock();
                convolve(&img, kernel, KERNEL_SIZE, &output_img);
                clock_t end_time = clock();
                double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

                // Write results to the markdown file
                FILE *f = fopen("serial_cpu_time.md", "a");
                if (f != NULL) {
                    int result = fprintf(f, "%s, %s, %f, %d\n", entry->d_name, kernel_names[i], elapsed_time, KERNEL_SIZE);
                    if (result < 0) {
                        perror("Error writing to file");
                    } else {
                        printf("Successfully wrote to file.\n");
                    }
                    fclose(f);
                } else {
                    perror("Error opening file");
                }

                // Free memory
                free_image_data(&output_img);
                free(kernel);
            }
        }
    }

    closedir(dir);

    return 0;
}
