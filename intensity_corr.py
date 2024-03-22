# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:49:13 2024

@author: annap
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.filters
import skimage.draw
from scipy.signal import find_peaks


def correct_intensity(image_path, threshold_factor=0.5):
    # Read the image
    image = skimage.io.imread(image_path, as_gray=True)

    # Compute the gradient of the image
    gradient = skimage.filters.sobel(image)

    # Get image dimensions
    height, width = image.shape

    # Calculate average gradient magnitude in the image
    avg_gradient = np.mean(gradient)

    # Create a copy of the original image for correction
    corrected_image = np.copy(image)

    # Iterate through each row in the image
    for y in range(height):
        row = image[y, :]
        grad_row = gradient[y, :]

        # Calculate the local gradient magnitude along the row
        local_gradient = np.abs(grad_row)

        # Determine threshold for local gradient based on the average gradient
        threshold = threshold_factor * avg_gradient

        # Find indices where the local gradient exceeds the threshold
        spike_indices = np.argwhere(local_gradient > threshold).flatten()

        # Correct intensities at spike locations
        for spike_index in spike_indices:
            # Get indices of neighboring pixels
            left_index = max(spike_index - 1, 0)
            right_index = min(spike_index + 1, width - 1)

            # Calculate corrected intensity
            corrected_intensity = (row[left_index] + row[right_index]) / 2

            # Apply correction
            corrected_image[y, spike_index] = corrected_intensity

    return corrected_image

# Example usage
image_path = 'NoNaNs.png'
threshold_factor = 0.5  # Adjust as needed

corrected_image = correct_intensity(image_path, threshold_factor)

# Display the original and corrected images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(skimage.io.imread(image_path), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(corrected_image, cmap='gray')
plt.title('Corrected Image')
plt.axis('off')

plt.show()


