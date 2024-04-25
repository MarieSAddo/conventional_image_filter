# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# from scipy.ndimage import convolve

# # Define a sharpening kernel
# kernel = np.array([[0, -1, 0],
#                    [-1, 5,-1],
#                    [0, -1, 0]])

# # Read the original image
# img_original = mpimg.imread('saddog.png')

# # Apply the kernel to the original image
# img_sharpened = convolve(img_original, kernel)

# # Save the sharpened image
# mpimg.imsave('output.png', img_sharpened)

# # Read the sharpened image
# img_sharpened = mpimg.imread('output.png')

# # Display the images side by side
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# # Display the original image
# axs[0].imshow(img_original, cmap='gray') 
# axs[0].set_title('Original Image')

# # Display the sharpened image
# axs[1].imshow(img_sharpened, cmap='gray')
# axs[1].set_title('Sharpened Image')

# # Label x-axis and y-axis of both images 
# # x-axis: horizontal axis is the width of the image
# # y-axis: vertical axis is the height of the image

# # Set labels for x-axis and y-axis
# plt.xlabel('Width')
# plt.ylabel('Height')

# plt.show()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read the original and sharpened images
img_original = mpimg.imread('saddog.png')
img_sharpened = mpimg.imread('output.png')

# Display the images side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
axs[0].imshow(img_original, cmap='gray') 
axs[0].set_title('Original Image')

print(img_sharpened.shape)

# Display the sharpened image
axs[1].imshow(img_sharpened, cmap='gray')
axs[1].set_title('Convoluted Image')

# Label x-axis and y-axis of both images 
# x-axis: horizontal axis is the width of the image
# y-axis: vertical axis is the height of the image

# Set labels for x-axis and y-axis
plt.xlabel('Width')
plt.ylabel('Height')

plt.show()