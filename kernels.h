#ifndef KERNELS_H
#define KERNELS_H

#include <stdlib.h>

/**
 * Returns a 2D Gaussian kernel of size `size` x `size` with the given standard deviation `sigma`.
 * The returned memory must be freed by the caller (a single free for the whole kernel).
 * The size should be an odd number, so that the kernel is centered.
 */
double** gauss_kernel(int size);
double** unsharpen_mask_kernel(int size);
double** mean_kernel(int size);

#endif /* KERNELS_H*/