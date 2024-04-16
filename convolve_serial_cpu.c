#include <stdio.h>
#include <stdlib.h>

// Function to perform convolution on a single pixel
float apply_kernel(float **image, float **kernel, int x, int y, int kernel_size) {
    float sum = 0.0;
    int offset = kernel_size / 2;

    for (int i = -offset; i <= offset; i++) {
        for (int j = -offset; j <= offset; j++) {
            sum += image[x + i][y + j] * kernel[offset + i][offset + j];
        }
    }

    return sum;
}

// Convolution function
void convolve_serial(float **image, float **output, float **kernel, int width, int height, int kernel_size) {
    int offset = kernel_size / 2;

    // Apply kernel to each pixel
    for (int i = offset; i < height - offset; i++) {
        for (int j = offset; j < width - offset; j++) {
            output[i][j] = apply_kernel(image, kernel, i, j, kernel_size);
        }
    }
}

int main() {
    int width, height, kernel_size;
    float **image, **output, **kernel;

    // Assume functions to read the image into 'image' buffer and kernel into 'kernel' buffer
    // read_image("path_to_image", &image, &width, &height);
    // read_kernel("path_to_kernel", &kernel, &kernel_size);

    // Allocate memory for the output buffer
    output = (float **)malloc(height * sizeof(float *));
    for (int i = 0; i < height; i++) {
        output[i] = (float *)malloc(width * sizeof(float));
    }

    // Perform convolution
    convolve(image, output, kernel, width, height, kernel_size);

    // Assume function to write the output buffer to an image file
    // write_image("path_to_output_image", output, width, height);

    // Free allocated memory
    for (int i = 0; i < height; i++) {
        free(image[i]);
        free(output[i]);
        free(kernel[i]);
    }
    free(image);
    free(output);
    free(kernel);

    return 0;
}
