# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:44:56 2024

@author: annap
"""
import cv2
import numpy as np
from PIL import Image

def correct_nan(image_path, iterations):
    # Open the multi-page TIFF file
    with Image.open(image_path) as img:
        # Iterate over each page/image in the TIFF file
        corrected_images = []
        for i in range(img.n_frames):
            img.seek(i)
            # Convert the current page/image to grayscale
            image = np.array(img.convert("L"))

            # Convert image to float32 for handling NaNs
            image_float = image.astype(np.float32)

            # Iterate for the desired number of iterations
            for _ in range(iterations):
                # Identify NaN regions (assuming NaN regions are completely black)
                nan_mask = (image_float == 0)

                # Find indices of NaN regions
                nan_indices = np.argwhere(nan_mask)

                # Iterate over NaN regions
                for idx in nan_indices:
                    x, y = idx

                    # Define neighborhood
                    neighborhood = image_float[max(0, x-1):min(x+2, image.shape[0]),
                                               max(0, y-1):min(y+2, image.shape[1])]

                    # Exclude NaN values from neighborhood
                    valid_pixels = neighborhood[~nan_mask[max(0, x-1):min(x+2, image.shape[0]),
                                                          max(0, y-1):min(y+2, image.shape[1])]]

                    # If there are valid pixels in the neighborhood, fill NaN pixel with mean of valid pixels
                    if len(valid_pixels) > 0:
                        image_float[x, y] = np.mean(valid_pixels)

            # Convert the corrected image back to uint8
            corrected_image = image_float.astype(np.uint8)
            corrected_images.append(corrected_image)

    return corrected_images

# Example usage
image_path = 'mode jump-demo.tif'
iterations = 5  # Adjust this value to set the number of iterations
corrected_images = correct_nan(image_path, iterations)

# Save each corrected image separately
output_base_path = "corrected_image_{}.tiff"
for i, corrected_image in enumerate(corrected_images):
    output_path = output_base_path.format(i)
    Image.fromarray(corrected_image).save(output_path)

# Combine corrected images into a multi-page TIFF file
with Image.open(output_base_path.format(0)) as first_img:
    first_img.save("corrected_multi_page_image.tiff", save_all=True, append_images=[Image.open(output_base_path.format(i)) for i in range(1, len(corrected_images))])

# Display the first corrected image
cv2.imshow('Corrected Image', corrected_images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()