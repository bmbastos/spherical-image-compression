from src.compressor import Compressor
from skimage.io import imread
from matplotlib import pyplot as plt

target_file = 'spherical_image_sample.bmp'
compressor = Compressor(imread('input_images/' + target_file, as_gray=True).astype(float))			

print(f"The image will be compressed by a quality factor of {compressor.get_quantization_factor()}."); print()
# Change the quality factor
# compressor.set_quantization_factor('Set here')

print(f"Transformation matrix:")
print(compressor.get_transformation_matrix()); print()
# compressor.set_transformation_matrix('Set here') # TB, TO, TR

print(f"Quantization matrix:")
print(compressor.get_quantization_matrix()); print()
# compressor.set_quantization_matrix('Set here') # Q0, QB, QHVS

original_image = compressor.get_image()
compressor.lower_complexity(); print()
reconstructed_image = compressor.get_image()

# If you wat to save the image
#plt.imsave('output_images/' + target_file.split(".")[0] + '.jpg', reconstructed_image, cmap='gray')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(reconstructed_image, cmap='gray')
axes[1].set_title('Compressed Image')
axes[1].axis('off')

fig.tight_layout()
plt.show()

