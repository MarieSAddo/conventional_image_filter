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

# Display the sharpened image
axs[1].imshow(img_sharpened, cmap='gray')
axs[1].set_title('Sharpened Image')

# Label x-axis and y-axis of both images 
# x-axis: horizontal axis is the width of the image
# y-axis: vertical axis is the height of the image

# Set labels for x-axis and y-axis
plt.xlabel('Width')
plt.ylabel('Height')

plt.show()