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

plt.show()