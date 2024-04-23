/**run: gcc-13 -o convolve_serial_cpu convolve_serial_cpu.c -lm && ./convolve_serial_cpu**/
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include "image.h"
#include <time.h>
#include "kernels.h"
#include <string.h>

#define KERNEL_SIZE 3

int clamp(double value, int min, int max) {
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return (int)value;
}

void convolve(Image *img, double** kernel, int kernel_size, Image *output_img)
{
    // Perform convolution
    for (int i = 0; i < img->height; i++) // Iterate over the rows of the image
    {
        for (int j = 0; j < img->width; j++)
        {
            // Initialize the output pixel value
            double output_pixel = 0;
            // Iterate over the kernel
            for (int k = 0; k < kernel_size; k++) // Iterate over the rows of the kernel
            {
                for (int l = 0; l < kernel_size; l++) // Iterate over the columns of the kernel
                {
                    // Calculate the coordinates of the pixel in the input image
                    int x_index = i + k - kernel_size / 2; 
                    int y_index = j + l - kernel_size / 2;
                    // Check if the pixel is within the bounds of the image
                    if (x_index >= 0 && x_index < img->height && y_index >= 0 && y_index < img->width)
                    {
                        // Multiply the kernel value with the corresponding pixel value in the input image
                        output_pixel += kernel[k][l] * (double)img->data[x_index][y_index];
                        
                    }
                }
            }
            // Set the output pixel value in the output image
            // Ensure the output_pixel value is within the range of pixel values
            output_img->data[i][j] = clamp(output_pixel, 0, 255); // You need to implement clamp function;
        }
    }
}

// void write_results_to_file(const char *filename, const char *results) {
//     FILE *file = fopen(filename, "a");
//     if (file == NULL) {
//         printf("Error opening file!\n");
//         return;
//     }

//     // Get the current time
//     time_t t = time(NULL);
//     struct tm *tm = localtime(&t);
//     char time_str[64];
//     strftime(time_str, sizeof(time_str), "%c", tm);

//     // Write the time and results to the file
//     fprintf(file, "## %s\n\n", time_str);
//     fprintf(file, "%s\n", results);

//     fclose(file);
// }

void free_kernel(double** kernel, int size) {
    free(kernel);
}

int main()
{
    int kernel_size = (rand() % 5) * 2 + 3;// Randomly generate a kernel size between 3 and 11 that is an odd number
    // Read the PNG file
    Image img;
    read_png_file("image1.png", PNG_COLOR_TYPE_GRAY, &img);

    // Prompt user to enter a kernel type they want to use
   char kernel_name[50];
    printf("Enter the type of kernel you want to use (gauss, unsharpen_mask, mean): ");    
    scanf("%s", kernel_name);

    //Generate kernel based on the user input
    double** kernel;
    
    if (strcmp(kernel_name, "gauss") == 0) {
        kernel = gauss_kernel(kernel_size);
    } else if (strcmp(kernel_name, "unsharpen_mask") == 0) {
        kernel = unsharpen_mask_kernel(kernel_size);
    } else if (strcmp(kernel_name, "mean") == 0) {
        kernel = mean_kernel(kernel_size);
    } else {
        printf("Invalid kernel type\n");
        return 1;
    }
    
    
    // Allocate memory for the output image
    Image output_img;
    output_img.width = img.width;
    output_img.height = img.height;
    output_img.color_type = PNG_COLOR_TYPE_GRAY;
    malloc_image_data(&output_img);

    // check if the data field of img and output_img are not null
    if (img.data == NULL || output_img.data == NULL)
    {
        printf("Error: Memory not allocated for image data\n");
        return 1;
    }


    // Perform convolution
    convolve(&img, kernel, kernel_size, &output_img);

    // Write the output image to a new JPEG file
    //write_png_file("output.png", &output_img);

    // write to an output file based on the kernel type
    char output_file[50];
    sprintf(output_file, "output_%s.png", kernel_name);
    write_png_file(output_file, &output_img);


    // Write the results to a results to a markdown file
    // write_results_to_file("results.md", "Here are the results...");

    // Free the memory allocated for the images
    free_image_data(&img);
    free_image_data(&output_img);
    free_kernel(kernel, kernel_size);
    return 0;
}