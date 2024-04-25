// /*This is a serial implementation of the convolution operation on an image using the CPU.
// This program reads an image from a PNG file, performs convolution on the image using a kernel of a specified type,
// and writes the output image to a new PNG file. It also writes the results to a serialized markdown file.
// *run: gcc-13 -o convolve_serial_cpu convolve_serial_cpu.c -lm && ./convolve_serial_cpu**/
// #include <stdio.h>
// #include <stdlib.h>
// #include <png.h>
// #include "image.h"
// #include <time.h>
// #include "kernels.h"
// #include <string.h>

// #define KERNEL_SIZE 3

// int clamp(double value, int min, int max) {
//     if (value < min)
//         return min;
//     else if (value > max)
//         return max;
//     else
//         return (int)value;
// }

// void convolve(Image *img, double** kernel, int kernel_size, Image *output_img)
// {
//     // Perform convolution
//     for (int i = 0; i < img->height; i++) // Iterate over the rows of the image
//     {
//         for (int j = 0; j < img->width; j++)
//         {
//             // Initialize the output pixel value
//             double output_pixel = 0;
//             // Calculate the coordinates of the pixel in the input image
//             int x_index = i - kernel_size / 2;
//             int y_index = j - kernel_size / 2;
//             // Iterate over the kernel
//             for (int k = 0; k < kernel_size; k++) // Iterate over the rows of the kernel
//             {
//                 for (int l = 0; l < kernel_size; l++) // Iterate over the columns of the kernel
//                 {
//                     // Check if the pixel is within the bounds of the image
//                     int x_temp = x_index + k;
//                     int y_temp = y_index + l;
//                     if (x_temp >= 0 && x_temp < img->height && y_temp >= 0 && y_temp < img->width)
//                     {
//                         // Multiply the kernel value with the corresponding pixel value in the input image
//                         output_pixel += kernel[k][l] * (double)img->data[x_temp][y_temp];
//                     }
//                 }
//             }
//             // Set the output pixel value in the output image
//             // Ensure the output_pixel value is within the range of pixel values
//             output_img->data[i][j] = (unsigned char)clamp(output_pixel, 0, 255);
//         }
//     }
// }

// void free_kernel(double** kernel) {
//     free(kernel);
// }

// int main()
// {
//     srand(0);
//     int kernel_size = (rand() % 5) * 2 + 3;// Randomly generate a kernel size between 3 and 11 that is an odd number
//     // Read the PNG file variable img
//     Image img;
//     read_png_file("images/waterfall_gr.png", PNG_COLOR_TYPE_GRAY, &img);

//     // Prompt user to enter a kernel type they want to use
//    char kernel_name[50];
//     printf("Enter the type of kernel you want to use (gauss, unsharpen_mask, mean): ");
//     scanf("%s", kernel_name);

//     //Generate kernel based on the user input
//     double** kernel;

//     if (strcmp(kernel_name, "gauss") == 0) {
//         kernel = gauss_kernel(kernel_size);
//     } else if (strcmp(kernel_name, "unsharpen_mask") == 0) {
//         kernel = unsharpen_mask_kernel(kernel_size);
//     } else if (strcmp(kernel_name, "mean") == 0) {
//         kernel = mean_kernel(kernel_size);
//     } else {
//         printf("Invalid kernel type\n");
//         return 1;
//     }

//     // Allocate memory for the output image
//     Image output_img;
//     output_img.width = img.width;
//     output_img.height = img.height;
//     output_img.color_type = PNG_COLOR_TYPE_GRAY;
//     malloc_image_data(&output_img);

//     // check if the data field of img and output_img are not null
//     if (img.data == NULL || output_img.data == NULL)
//     {
//         printf("Error: Memory not allocated for image data\n");
//         return 1;
//     }

//     // start the clock
//     clock_t start_time = clock();

//     // Perform convolution
//     convolve(&img, kernel, kernel_size, &output_img);

//     // get the end time and calculate the elapsed time
//     clock_t end_time = clock();
//     double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

//      FILE *f = fopen("serial_cpu_time.md", "a");
//      // print header row
//      // serial, kernel type, time, kernel size

//     if (f != NULL)
//     {
//         fprintf(f, "Serial, %s, %f, %d\n", kernel_name, elapsed_time, kernel_size);
//         fclose(f);
//     }
//     else
//     {
//         printf("Error opening file!\n");
//     }
//     fclose(f);

//     // Write the output image to a new JPEG file
//     //write_png_file("output.png", &output_img);

//     // write to an output file based on the serial, kernel type, image and kernel size
//     char output_file[128];
//     sprintf(output_file, "output_%s.png", kernel_name);
//     write_png_file(output_file, &output_img);

//     // Write the results to a results to a markdown file
//     // write_results_to_file("results.md", "Here are the results...");

//     // Free the memory allocated for the images
//     free_image_data(&img);
//     free_image_data(&output_img);
//     free_kernel(kernel);
//     return 0;
// }
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h> // For directory handling
#include <string.h>
#include <png.h>
#include "image.h"
#include <time.h>
#include "kernels.h"

#define KERNEL_SIZE 3

int clamp(double value, int min, int max)
{
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return (int)value;
}

void convolve(Image *img, double **kernel, int kernel_size, Image *output_img)
{
    // Perform convolution
    for (int i = 0; i < img->height; i++) // Iterate over the rows of the image
    {
        for (int j = 0; j < img->width; j++)
        {
            // Initialize the output pixel value
            double output_pixel = 0;
            // Calculate the coordinates of the pixel in the input image
            int x_index = i - kernel_size / 2;
            int y_index = j - kernel_size / 2;
            // Iterate over the kernel
            for (int k = 0; k < kernel_size; k++) // Iterate over the rows of the kernel
            {
                for (int l = 0; l < kernel_size; l++) // Iterate over the columns of the kernel
                {
                    // Check if the pixel is within the bounds of the image
                    int x_temp = x_index + k;
                    int y_temp = y_index + l;
                    if (x_temp >= 0 && x_temp < img->height && y_temp >= 0 && y_temp < img->width)
                    {
                        // Multiply the kernel value with the corresponding pixel value in the input image
                        output_pixel += kernel[k][l] * (double)img->data[x_temp][y_temp];
                    }
                }
            }
            // Set the output pixel value in the output image
            // Ensure the output_pixel value is within the range of pixel values
            output_img->data[i][j] = (unsigned char)clamp(output_pixel, 0, 255);
        }
    }
}

void free_kernel(double **kernel)
{
    free(kernel);
}

int main()
{
    srand(0);

    // Open the "images" directory
    DIR *dir = opendir("images");
    if (dir == NULL)
    {
        printf("Error opening images directory\n");
        return 1;
    }

    // Array to store kernel names (assuming a limited number of kernels)
    char kernel_names[3][50] = {"gauss", "unsharpen_mask", "mean"};

    // Loop through all files in the directory
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        {
            continue; // Skip "." and ".." entries
        }

        // Check if the entry is a regular file (image)
        if (entry->d_type & DT_REG)
        {
            char image_path[128];
            sprintf(image_path, "images/%s", entry->d_name);

            // Read the image
            Image img;
            read_png_file(image_path, PNG_COLOR_TYPE_GRAY, &img);

            // Loop through each kernel type
            for (int i = 0; i < 3; i++)
            {
                double **kernel = NULL;
                if (strcmp(kernel_names[i], "gauss") == 0)
                {
                    kernel = gauss_kernel(KERNEL_SIZE);
                }
                else if (strcmp(kernel_names[i], "unsharpen_mask") == 0)
                {
                    kernel = unsharpen_mask_kernel(KERNEL_SIZE);
                }
                else if (strcmp(kernel_names[i], "mean") == 0)
                {
                    kernel = mean_kernel(KERNEL_SIZE);
                }

                // Allocate memory for the output image
                Image output_img;
                output_img.width = img.width;
                output_img.height = img.height;
                output_img.color_type = PNG_COLOR_TYPE_GRAY;
                malloc_image_data(&output_img);

                // Perform convolution
                clock_t start_time = clock();
                convolve(&img, kernel, KERNEL_SIZE, &output_img);
                clock_t end_time = clock();
                double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

                // Write results to the markdown file
                // Write results to the markdown file
                FILE *f = fopen("serial_cpu_time.md", "a");
                if (f != NULL)
                {
                    int result = fprintf(f, "%s, %s, %f, %d\n", entry->d_name, kernel_names[i], elapsed_time, KERNEL_SIZE);
                    if (result < 0)
                    {
                        perror("Error writing to file");
                    }
                    else
                    {
                        printf("Successfully wrote to file.\n");
                    }
                    fclose(f);
                }
                else
                {
                    perror("Error opening file");
                }
                // FILE *f = fopen("serial_cpu_time.md", "a");
                // if (f != NULL) {
                //     fprintf(f, "%s, %s, %f, %d\n", entry->d_name, kernel_names[i], elapsed_time, KERNEL_SIZE);
                //     fclose(f);
                // } else {
                //     printf("Error opening file!\n");
                // }

                // // Write the output image (optional)
                // char output_file[128];
                // sprintf(output_file, "output_%s_%s.png", entry->d_name, kernel_names[i]);
                // write_png_file(output_file, &output_img);

                // Free memory
                free_image_data(&output_img);
                free_kernel(kernel);
            }
        }
    }

    closedir(dir);

    return 0;
}