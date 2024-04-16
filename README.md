# conventional_image_filter
cs392 Project Final

# Contributers
[Marie] (https://github.com/MarieSAddo)\
[Jeremy] 


# Resources
https://github.com/jimouris/parallel-convolution
https://github.com/topics/parallel-processing?l=c
https://github.com/topics/image-convolution
https://stackoverflow.com/questions/58926559/blurring-an-image-with-c
https://learnopencv.com/image-filtering-using-convolution-in-opencv/
https://stackoverflow.com/questions/58252334/convolution-for-image-filtering
https://medium.com/@henriquevedoveli/image-filtering-techniques-in-image-processing-part-1-d03362fc73b7
https://ulhpc-tutorials.readthedocs.io/en/latest/cuda/exercises/convolution/

# Will add Project pdf to the project 

Serial with CPU: assumes that you have a function to read the image 
into a 2D array and a function to write the output image from a 2D array. The 
convolution operation is performed by a function convolve_serial which 
takes the 
image, kernel, and output buffers as arguments along with their 
dimensions.

Parallel with CPU using OpenMP: uses OpenMP and assumes that you have a 
function to read the image into a 2D array and a function to write the output image 
from a 2D array. The convolution operation is performed by a function 
convolve_parallel which takes the image, kernel, and output buffers as 
arguments along with their dimensions.

Parallel GPU with CUDA: uses CUDA the convolve_gpu function sets up the 
memory and calls the kernel. The main function demonstrates how to 
allocate memory, initialize data, and call the GPU convolution function.
