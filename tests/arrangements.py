from src.compressor import Compressor
from src.matrices import *
from skimage.io import imread
from matplotlib import pyplot as plt
from pdb import set_trace as pause

target_file = 'spherical_image_sample.bmp'
original_image = Compressor.process_image(imread('input_images/' + target_file, as_gray=True).astype(float))

compressor_1 = Compressor(original_image)
compressor_2 = Compressor(original_image)

print(f"The image will be compressed by a quality factor of {compressor_1.get_quantization_factor()}."); print()
# Change the quality factor
# compressor.set_quantization_factor('Set here')

print(f"Transformation matrix:")
print(compressor_1.get_transformation_matrix()); print()
# compressor.set_transformation_matrix('Set here') # TB, TO, TR

print(f"Quantization matrix:")
print(compressor_1.get_quantization_matrix()); print()
# compressor.set_quantization_matrix('Set here') # Q0, QB, QHVS

compressor_1.lower_complexity_2(original_image); print()
compressor_2.matematically_correct(original_image); print()
reconstructed_image_1 = compressor_1.get_image()
reconstructed_image_2 = compressor_2.get_image()

fig, axes = plt.subplots(3, 1, figsize=(10, 5))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(reconstructed_image_1, cmap='gray')
axes[1].set_title('Compressed Image with lower_complexity_2')
axes[1].axis('off')

axes[2].imshow(reconstructed_image_2, cmap='gray')
axes[2].set_title('Compressed Image with matematically_correct')
axes[2].axis('off')

fig.tight_layout()
#plt.show()

ws_psnr_1 = Compressor.calculate_wpsnr(original_image, reconstructed_image_1)
ws_ssim_1 = Compressor.calculate_wssim(original_image, reconstructed_image_1); print()

ws_psnr_2 = Compressor.calculate_wpsnr(original_image, reconstructed_image_2)
ws_ssim_2 = Compressor.calculate_wssim(original_image, reconstructed_image_2); print()


print(f"WS-PSNR of the first compressed image: {ws_psnr_1}"); print(f"WS-PSNR of the second compressed image: {ws_psnr_2}"); print()
print(f"WS-SSIM of the first compressed image: {ws_ssim_1}"); print(f"WS-SSIM of the second compressed image: {ws_ssim_2}"); print()
print(f"BPP of the first compressed image: {compressor_1.get_bpp()}"); print(f"BPP of the second compressed image: {compressor_2.get_bpp()}")


