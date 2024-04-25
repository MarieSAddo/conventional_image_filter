// /* This file will run with the GPU in parallel*/
// #include <stdio.h>
// #include <cuda_runtime.h>

// // CUDA kernel for performing convolution on the GPU
// __global__ void convolve_kernel(float *input, float *output, float *kernel, int width, int height, int kernel_size) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int offset = kernel_size / 2;

//     if (x >= offset && x < width - offset && y >= offset && y < height - offset) {
//         float sum = 0.0;
//         for (int i = -offset; i <= offset; i++) {
//             for (int j = -offset; j <= offset; j++) {
//                 sum += input[(y + i) * width + (x + j)] * kernel[(i + offset) * kernel_size + (j + offset)];
//             }
//         }
//         output[y * width + x] = sum;
//     }
// }

// // Host function to run the convolution on the GPU
// void convolve_gpu(float *h_input, float *h_output, float *h_kernel, int width, int height, int kernel_size) {
//     float *d_input, *d_output, *d_kernel;
//     size_t bytes = width * height * sizeof(float);

//     // Allocate memory on the GPU
//     cudaMalloc((void **)&d_input, bytes);
//     cudaMalloc((void **)&d_output, bytes);
//     cudaMalloc((void **)&d_kernel, kernel_size * kernel_size * sizeof(float));

//     // Copy data from host to device
//     cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

//     // Define block size and grid size
//     dim3 block_size(16, 16);
//     dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

//     // Launch the kernel
//     convolve_kernel<<<grid_size, block_size>>>(d_input, d_output, d_kernel, width, height, kernel_size);

//     // Copy result back to host
//     cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

//     // Free memory on GPU
//     cudaFree(d_input);
//     cudaFree(d_output);
//     cudaFree(d_kernel);
// }

// int main() {
//     // Image dimensions
//     int width = 1024; // example width
//     int height = 1024; // example height
//     int kernel_size = 3; // example kernel size

//     // Allocate memory for the image and kernel on the host
//     float *h_input = (float *)malloc(width * height * sizeof(float));
//     float *h_output = (float *)malloc(width * height * sizeof(float));
//     float *h_kernel = (float *)malloc(kernel_size * kernel_size * sizeof(float));

//     // Initialize image and kernel here
//     // ...

//     // Run the convolution on the GPU
//     convolve_gpu(h_input, h_output, h_kernel, width, height, kernel_size);

//     // Use the result in h_output
//     // ...

//     // Free host memory
//     free(h_input);
//     free(h_output);
//     free(h_kernel);

//     return 0;
// }
