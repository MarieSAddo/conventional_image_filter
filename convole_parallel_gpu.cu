#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h> // Include for malloc and free
#include <string.h> // Include for strcmp
#include <sys/time.h> // Include for timing

// CUDA Kernel for convolution
__global__ void convolutionKernel(int *inputImage, int *outputImage, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int sum = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int xIndex = col + i;
                int yIndex = row + j;
                // Check boundary conditions
                if (xIndex >= 0 && yIndex >= 0 && xIndex < width && yIndex < height) {
                    sum += inputImage[yIndex * width + xIndex];
                }
            }
        }
        outputImage[row * width + col] = sum / 9; // Assuming a 3x3 kernel
    }
}

// Host function to run the convolution on the GPU
void convolve_gpu(int *inputImage, int *outputImage, int width, int height, int (*kernel)(int*, int, int, int, int)) {
    int imageSize = width * height * sizeof(int);

    // Allocate memory for the input and output images
    int *d_inputImage, *d_outputImage;
    cudaMalloc((void **)&d_inputImage, imageSize);
    cudaMalloc((void **)&d_outputImage, imageSize);

    // Copy input image to device memory
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Start timer
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Launch convolution kernel
    convolutionKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);
    cudaDeviceSynchronize();

    // End timer
    gettimeofday(&end, NULL);
    float elapsed = (end.tv_sec - start.tv_sec) * 1000.0; // sec to ms
    elapsed += (end.tv_usec - start.tv_usec) / 1000.0;    // us to ms

    // Copy output image from device memory to host memory
    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    printf("Convolution completed in %.2f milliseconds.\n", elapsed);
}

// Example Gaussian kernel
__device__ int gauss_kernel(int *inputImage, int col, int row, int width, int height) {
    int sum = 0;
    // Placeholder implementation
    return sum;
}

// Example unsharp mask kernel
__device__ int unsharpen_mask_kernel(int *inputImage, int col, int row, int width, int height) {
    int sum = 0;
    // Placeholder implementation
    return sum;
}

// Example mean kernel
__device__ int mean_kernel(int *inputImage, int col, int row, int width, int height) {
    int sum = 0;
    // Placeholder implementation
    return sum;
}

int main() {
    int width = 1024; // Width of the image
    int height = 1024; // Height of the image
    int imageSize = width * height * sizeof(int);

    // Allocate memory for the input and output images
    int *inputImage, *outputImage;
    inputImage = (int*)malloc(imageSize);
    outputImage = (int*)malloc(imageSize);

    // Initialize input image with random values
    for (int i = 0; i < width * height; i++) {
        inputImage[i] = rand() % 256; // Assuming pixel values are between 0 and 255
    }

    // Prompt user to choose kernel type
    char kernel_name[50];
    printf("Enter the type of kernel you want to use (gauss, unsharpen_mask, mean): ");
    scanf("%s", kernel_name);

    // Run convolution on the GPU
    if (strcmp(kernel_name, "gauss") == 0) {
        // Call function with Gaussian kernel
        convolve_gpu(inputImage, outputImage, width, height, gauss_kernel);
    } else if (strcmp(kernel_name, "unsharpen_mask") == 0) {
        // Call function with unsharp mask kernel
        convolve_gpu(inputImage, outputImage, width, height, unsharpen_mask_kernel);
    } else if (strcmp(kernel_name, "mean") == 0) {
        // Call function with mean kernel
        convolve_gpu(inputImage, outputImage, width, height, mean_kernel);
    } else {
        printf("Invalid kernel name. Exiting...\n");
        return -1;
    }

    // Free allocated memory
    free(inputImage);
    free(outputImage);

    return 0;
}
