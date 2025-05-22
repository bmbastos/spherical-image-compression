from src.compressor import Compressor
from src.matrices import *
from skimage.io import imread
from matplotlib import pyplot as plt

target_file = 'spherical_image_sample.bmp'
original_image = Compressor.process_image(imread('input_images/' + target_file, as_gray=True).astype(float))

compressor = Compressor(original_image)

print(f"The image will be compressed by a quality factor of {compressor.get_quantization_factor()}."); print()
# Change the quality factor
# compressor.set_quantization_factor('Set here')

compressed_image = compressor.lower_complexity(original_image); print()
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

print(f"WS-PSNR of the compressed image: {Compressor.calculate_wpsnr(original_image, reconstructed_image)}"); print()
print(f"WS-SSIM of the compressed image: {Compressor.calculate_wssim(original_image, reconstructed_image)}"); print()
print(f"BPP of the compressed image: {Compressor.calculate_bpp(compressed_image)}"); print()

fig.tight_layout()
plt.show()

