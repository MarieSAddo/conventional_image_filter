#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h> // Include for malloc and free

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
void convolve_gpu(int *inputImage, int *outputImage, int width, int height) {
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

    // Launch convolution kernel
    convolutionKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);
    cudaDeviceSynchronize();

    // Copy output image from device memory to host memory
    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
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

    // Run convolution on the GPU
    convolve_gpu(inputImage, outputImage, width, height);

    // Output the result
    printf("Convolution completed.\n");

    // Free allocated memory
    free(inputImage);
    free(outputImage);

    return 0;
}
