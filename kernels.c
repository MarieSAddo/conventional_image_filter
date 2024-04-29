#include <math.h>
#include <stdlib.h>

/**
 * Returns a 2D Gaussian kernel of size `size` x `size` with the given standard deviation `sigma`.
 * The returned memory must be freed by the caller (a single free for the whole kernel).
 * The size should be an odd number, so that the kernel is centered.
 */
double** gauss_kernel(int size) {
    double** gauss = (double**)malloc(size*sizeof(double*) + size*size*sizeof(double));
    double sum = 0;
    double sigma = (size - 1) / 8.0; // radius is 4 * standard deviation
    double factor = -1/(2*sigma*sigma);
    for (int i = 0; i < size; i++) {
        gauss[i] = (double*)(gauss + size) + i*size;
        for (int j = 0; j < size; j++) {
            double x = i - (size - 1) / 2.0;
            double y = j - (size - 1) / 2.0;
            gauss[i][j] = exp((x*x + y*y) * factor);
            sum += gauss[i][j];
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            gauss[i][j] /= sum;
        }
    }
    return gauss;
}


/**
 * Returns a 2D unsharp mask kernel of size `size` x `size`.
 * The returned memory must be freed by the caller (a single free for the whole kernel).
 * The size should be an odd number, so that the kernel is centered.
 */
double** unsharpen_mask_kernel(int size) {
    double** mask = gauss_kernel(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mask[i][j] = -mask[i][j];
        }
    }
    mask[(size-1)/2][(size-1)/2] += 2;
    return mask;
}


/**
 * Returns a 2D mean filter kernel of size `size` x `size`. Also known as box filter.
 * The returned memory must be freed by the caller (a single free for the whole kernel).
 * The size should be an odd number, so that the kernel is centered.
 */
double** mean_kernel(int size) {
    double** mean = (double**)malloc(size*sizeof(double*) + size*size*sizeof(double));
    for (int i = 0; i < size; i++) {
        mean[i] = (double*)(mean + size) + i*size;
        for (int j = 0; j < size; j++) {
            mean[i][j] = 1.0 / (size*size);
        }
    }
    return mean;
}

// // Testing main
// #include <stdio.h>
// int main() {
//     int size = 5;
//     printf("Gaussian kernel of size %d x %d:\n\n", size, size);
//     double** gauss = gauss_kernel(size);
//     for (int i = 0; i < size; i++) {
//         for (int j = 0; j < size; j++) {
//             printf("%f ", gauss[i][j]);
//         }
//         printf("\n");
//     }
//     free(gauss);
//     printf("\n");

//     printf("Unsharpen mask kernel of size %d x %d:\n\n", size, size);
//     double** mask = unsharpen_mask_kernel(size);
//     for (int i = 0; i < size; i++) {
//         for (int j = 0; j < size; j++) {
//             printf("%f ", mask[i][j]);
//         }
//         printf("\n");
//     }
//     free(mask);
//     printf("\n");

//     printf("Mean kernel of size %d x %d:\n\n", size, size);
//     double** mean = mean_kernel(size);
//     for (int i = 0; i < size; i++) {
//         for (int j = 0; j < size; j++) {
//             printf("%f ", mean[i][j]);
//         }
//         printf("\n");
//     }
//     free(mean);
    
//     return 0;
// }