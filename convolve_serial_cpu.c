/**run: gcc-13 -o convolve_serial_cpu convolve_serial_cpu.c -lm && ./convolve_serial_cpu**/
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include "image.h"
#include <time.h>

#define KERNEL_SIZE 3

int clamp(double value, int min, int max) {
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return (int)value;
}

void convolve(Image *img, double kernel [KERNEL_SIZE][KERNEL_SIZE], int kernel_size, Image *output_img)
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
                        output_pixel += kernel[k][l] * img->data[x_index][y_index];
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

int main()
{
    // Read the PNG file
    Image img;
    read_png_file("image1.png", PNG_COLOR_TYPE_GRAY, &img);

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


    // Define the sharpening kernel
    double kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}};


    // Perform convolution
    convolve(&img, kernel, KERNEL_SIZE, &output_img);

    // Write the output image to a new JPEG file
    write_png_file("output.png", &output_img);

    // Write the results to a results to a markdown file
    // write_results_to_file("results.md", "Here are the results...");

    // Free the memory allocated for the images
    free_image_data(&img);
    free_image_data(&output_img);
    return 0;
}