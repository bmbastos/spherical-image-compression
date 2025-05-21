from src.compressor import Compressor
from src.matrices import *
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from matplotlib import pyplot as plt
from pdb import set_trace as pause
from tqdm import tqdm
import csv
import os

datas = dict()

methods = ['JPEG', 'De Simone', 'Oliveira', 'Brahimi', 'Araar', 'Raiza', 'Lower Complexity 1', 'Lower Complexity 2', 'Mathematically Correct']

files = os.listdir('input_images/') # Change the path to the input images folder
quality_factors = list(range(5, 100, 5))

# Número total de iterações
total_steps = len(files) * len(methods) * len(quality_factors)
progress_bar = tqdm(total=total_steps, desc="Compressing all images")

for file in files:
	#print(file)
	original_image = Compressor.process_image(imread('input_images/' + file, as_gray=True).astype(float))

	compressed_image = original_image.copy()
	datas[file] = {}
	for method in methods:
		datas[file][method] = {}
		for qf in range(5, 100, 5):
			progress_bar.set_description(f"Processing {file} with {method} at QF {qf}")
			datas[file][method][qf] = {}
			compressor = Compressor(original_image)
			compressor.set_quantization_factor(qf)
			if method == 'JPEG': compressed_image = compressor.standard_jpeg(original_image)
			elif method == 'De Simone': compressed_image = compressor.proposed_from_simone(original_image)
			elif method == 'Oliveira': compressed_image = compressor.proposed_from_oliveira(original_image)
			elif method == 'Brahimi': compressed_image = compressor.proposed_from_brahimi(original_image)
			elif method == 'Araar': compressed_image = compressor.proposed_from_araar(original_image)
			elif method == 'Raiza': compressed_image = compressor.proposed_from_raiza(original_image)
			elif method == 'Lower Complexity 1': compressed_image = compressor.lower_complexity(original_image)
			elif method == 'Lower Complexity 2': compressed_image = compressor.lower_complexity_2(original_image)
			elif method == 'Mathematically Correct': compressed_image = compressor.matematically_correct(original_image)
		
			reconstructed_image = compressor.get_image()

			psnr_ = Compressor.calculate_wpsnr(original_image, reconstructed_image)
			ssim_ = Compressor.calculate_wssim(original_image, reconstructed_image)
			bpp_ = Compressor.calculate_bpp(compressed_image)

			datas[file][method][qf]['ws_psnr'] = psnr_
			datas[file][method][qf]['ws_ssim'] = ssim_
			datas[file][method][qf]['bpp'] = bpp_

			progress_bar.update(1)

destination = os.getcwd() + '/outputs/csv_files/'
fieldnames = ['File', 'Method', 'Quality Factor', 'WS-PSNR', 'WS-SSIM', 'BPP']
with open(destination + 'benchmark_results.csv', 'w', newline='') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for file in datas:
		for method in datas[file]:
			for qf in datas[file][method]:
				writer.writerow({
					'File': file,
					'Method': method,
					'Quality Factor': qf,
					'WS-PSNR': datas[file][method][qf]['ws_psnr'],
					'WS-SSIM': datas[file][method][qf]['ws_ssim'],
					'BPP': datas[file][method][qf]['bpp']
				})




















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
