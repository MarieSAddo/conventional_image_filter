#include <stdio.h>
#include <stdlib.h>
#include <dirent.h> // For directory handling
#include <string.h>
#include <png.h>
#include "image.h"
#include <time.h>
#include <cuda_runtime.h>
#include <kernel.h>

#define KERNEL_SIZE 3

// CUDA kernel for convolution - processes a single image-kernel combination
__global__ void convolutionKernel(const unsigned char* image_data, const double* kernel, 
                                    int image_width, int image_height, int image_index, 
                                    int kernel_index, unsigned char* output_data) {
    // Calculate thread and image coordinates based on block/grid layout and index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure thread is within image bounds
    if (idx < image_width && idy < image_height) {
        // Calculate output pixel index
        int output_index = idy * image_width + idx;

        // Initialize output pixel value
        double output_pixel = 0;
        
        // Iterate over the kernel
        for (int k = 0; k < KERNEL_SIZE; k++) {
            for (int l = 0; l < KERNEL_SIZE; l++) {
                // Check if pixel is within image bounds (considering kernel size)
                int x_temp = idy - KERNEL_SIZE / 2 + k;
                int y_temp = idx - KERNEL_SIZE / 2 + l;
                if (x_temp >= 0 && x_temp < image_height && y_temp >= 0 && y_temp < image_width) {
                    // Calculate image data index
                    int image_index = x_temp * image_width + y_temp;
                    // Perform convolution calculation
                    output_pixel += kernel[k * KERNEL_SIZE + l] * (double)image_data[image_index];
                }
            }
        }
        
        // Clamp and set output pixel value
        output_data[output_index] = (unsigned char)clamp(output_pixel, 0, 255);
    }
}

// Launches multiple convolution kernels for each image and kernel combination
void parallel_convolve(Image *images, int num_images, double** kernels, int num_kernels, 
                         Image* output_images) {
    // Allocate and transfer data to GPU
    unsigned char* d_images;
    cudaMalloc((void**)&d_images, num_images * images[0].width * images[0].height * sizeof(unsigned char)); // Assuming all images have the same size and type (grayscale)
    cudaMemcpy(d_images, images[0].data, num_images * images[0].width * images[0].height * sizeof(unsigned char), cudaMemcpyHostToDevice); // Assuming all images have the same size and type (grayscale) 

    double* d_kernels; // Assuming all kernels have the same size (3x3)
    int total_kernel_size = num_kernels * KERNEL_SIZE * KERNEL_SIZE; // Total size of all kernels combined (assuming same size)
    cudaMalloc((void**)&d_kernels, total_kernel_size * sizeof(double)); // Allocate memory for all kernels on the GPU (assuming same size)
    cudaMemcpy(d_kernels, kernels[0], total_kernel_size * sizeof(double), cudaMemcpyHostToDevice); // Transfer all kernels to the GPU (assuming same size) 

    unsigned char* d_output_images; // Allocate memory for output images on the GPU (assuming same size)
    cudaMalloc((void**)&d_output_images, num_images * images[0].width * images[0].height * sizeof(unsigned char)); // Allocate memory for output images on the GPU (assuming same size)     

    // Define grid size based on number of images and kernels
    dim3 gridSize(num_images, num_kernels); // Grid size based on number of images and kernels (assuming 2D grid)
    
    // Launch convolution kernel for each image-kernel combination
    convolutionKernel<<<gridSize, blockDim.x * blockDim.y>>>( // Launch kernel with 2D grid and 2D block
        d_images, d_kernels, images[0].width, images[0].height, 0, 0, d_output_images); // Assuming single image and kernel for simplicity (modify as needed)

    // Transfer results back and free memory
    cudaMemcpy(output_images[0].data, d_output_images, num_images * images[0].width * images[0].height * sizeof(unsigned char), cudaMemcpyDeviceToHost); // Transfer output image data back to host (assuming same size)
    cudaFree(d_images);
    cudaFree(d_kernels);
    cudaFree(d_output_images);
}

int main() {
    srand(0);

    // Open the "
// Open the "images" directory (unchanged)
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

        // Read the image (unchanged)
        Image img;
        read_png_file(image_path, PNG_COLOR_TYPE_GRAY, &img);

        // Allocate memory for output images (array to store processed versions)
        int num_kernels = 3; // Assuming you have 3 kernels
        Image* output_images = (Image*)malloc(num_kernels * sizeof(Image));
        for (int i = 0; i < num_kernels; i++) {
            output_images[i].width = img.width;
            output_images[i].height = img.height;
            output_images[i].color_type = PNG_COLOR_TYPE_GRAY;
            malloc_image_data(&output_images[i]);
        }

        // Load kernels (assuming you have functions to load them)
        double* kernels[num_kernels];
        for (int i = 0; i < num_kernels; i++) {
            kernels[i] = load_kernel(kernel_names[i], KERNEL_SIZE);
        }

        // Perform parallel convolution for all kernels on the image
        parallel_convolve(&img, 1, kernels, num_kernels, output_images);

        // Write results to separate files (assuming appropriate naming)
        for (int i = 0; i < num_kernels; i++) {
            char output_path[128];
            sprintf(output_path, "output/%s_%s.png", entry->d_name, kernel_names[i]);
            write_png_file(output_path, &output_images[i]);
            free_image_data(&output_images[i]);
            free(kernels[i]);
        }
        free(output_images);
    }
}

closedir(dir);

return 0;
}