from src.compressor import Compressor
from src.matrices import *
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from matplotlib import pyplot as plt
from pdb import set_trace as pause
import os

datas = dict()

files = os.listdir('input_images/') # Change the path to the input images folder

for file in files:
	if file.endswith('.bmp'): continue
	print(file)
	original_image = Compressor.process_image(imread('input_images/' + file, as_gray=True).astype(float))
	ws_psnr, ws_ssim, bpp = [], [], []

	for qf in range(5, 100, 5):
		print('-- QF =', qf, '--')
		compressor = Compressor(original_image)

		compressor.set_quantization_factor(qf)
		compressed_image = compressor.proposed_from_brahimi(original_image)
		reconstructed_image = compressor.get_image()

		#print(reconstructed_image)
		#pause()

		psnr_ = Compressor.calculate_wpsnr(original_image, reconstructed_image)
		ssim_ = Compressor.calculate_wssim(original_image, reconstructed_image)
		bpp_ = Compressor.calculate_bpp(compressed_image)

		print(f'PSNR: {peak_signal_noise_ratio(original_image, reconstructed_image, data_range=255)}')
		print(f'SSIM: {structural_similarity(original_image, reconstructed_image, data_range=255)}')
		#ws_psnr.append(psnr_)
		#ws_ssim.append(ssim_)
		bpp.append(bpp_)

		#print(f"WS-PSNR: {psnr_}")
		#print(f"WS-SSIM: {ssim_}")
		print(f"BPP: {bpp_}"); print()
		"""
		fig, axes = plt.subplots(3, 1, figsize=(10, 5))
		axes[0].imshow(original_image, cmap='gray')
		axes[0].set_title('Original Image')
		axes[0].axis('off')

		axes[1].imshow(compressed_image, cmap='gray')
		axes[1].set_title(f'Compressed Image with QF = {qf}')
		axes[1].axis('off')

		axes[2].imshow(reconstructed_image, cmap='gray')
		axes[2].set_title(f'Reconstructed Image with QF = {qf}')
		axes[2].axis('off')

		fig.tight_layout()
		plt.show()
		
	datas[file] = {'ws_psnr': ws_psnr, 'ws_ssim': ws_ssim, 'bpp': bpp}

plt.clf()
plt.cla()
plt.plot(datas['4.2.03.tiff']['bpp'] ,datas['4.2.03.tiff']['ws_psnr'], label='WS-PSNR')
plt.show()
plt.plot(datas['4.2.03.tiff']['bpp'] ,datas['4.2.03.tiff']['ws_ssim'], label='WS-SSIM')
plt.show()
pause()"""



















"""
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

"""
