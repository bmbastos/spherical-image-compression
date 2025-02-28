import os
import sys
import random
import csv
from numpy import array, zeros, ceil
import matplotlib.pyplot as plt
from matplotlib import colors

def pre_processing(csv_file_name: str) -> tuple:
	data_set = []
	methods = []
	with open(csv_file_name, 'r') as file:
		reader = csv.DictReader(file)
		for row in reader:
			file_name, method, psnr, ssim, bpp = row.values()
			methods.append(method)
			data_set.append({
				'Filename': file_name,
				'Method': method,
				'PSNR': list(map(float, psnr.strip('[]').split(','))),
				'SSIM': list(map(float, ssim.strip('[]').split(','))),
				'BPP': list(map(float, bpp.strip('[]').split(',')))
			})
	methods = list(set(methods))
	return data_set, methods

def random_unique_colors(num_colors):
	colors_css4 = list(colors.CSS4_COLORS.values())
	dark_colors = [color for color in colors_css4 if colors.to_rgb(color)[2] < 0.5]
	unique_colors = set()
	while len(unique_colors) < num_colors:
		random_color = random.choice(dark_colors)
		unique_colors.add(random_color)
	return list(unique_colors)

def averages(data_set: list, methods: list) -> dict:
	n_images = len(data_set) // len(methods)
	avgs = {}
	print(f'Quantidade de imagens {n_images}')
	for method in methods:
		avg_psnr = zeros(19, dtype=float)
		avg_ssim = zeros(19, dtype=float)
		avg_bpp = zeros(19, dtype=float)
		for data in data_set:
			if data['Method'] == method:
				avg_psnr += array(data['PSNR'])
				avg_ssim += array(data['SSIM'])
				avg_bpp += array(data['BPP'])

		if method.startswith('JPEG'):
			color, style = 'black', 'solid'
			method = r'JPEG [6]'
		elif method.startswith('OLIVEIRA'):
			color, style = 'red', 'dotted'
			method = r'Oliveira et al. [7]'
		elif method.startswith('BRAHIMI'):
			color, style = 'green', 'dotted'
			method = r'Brahimi et al. [8]'
		elif method.startswith('RAIZA'):
			color, style = 'magenta', 'dotted'
			method = r'Raiza et al. [9]'
		elif method.startswith('DE_SIMONE'):
			color, style = 'blue', 'dotted'
			method = r'De Simone et al. [2]'


		avgs[method] = {
			'PSNR': avg_psnr / n_images,
			'SSIM': avg_ssim / n_images,
			'BPP': avg_bpp / n_images,
			'Color': color,
			'Style': style
		}
	return avgs

# __MAIN__#
target_file = "original_proposes.csv"
dataset, methods = pre_processing(target_file)
methods.sort()
methods.insert(0, methods.pop(methods.index("JPEG")))
avgs = averages(dataset, methods)

max_psnr = round(max(max(avgs[avg]['PSNR']) for avg in avgs)) + 3
max_bpp = ceil(max(max(avgs[avg]['BPP']) for avg in avgs))
min_psnr = round(min(min(avgs[avg]['PSNR']) for avg in avgs)) - 3
min_ssim = min(min(avgs[avg]['SSIM']) for avg in avgs) - 0.03

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 7
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (3.4, 2.55)

# Plot WS-PSNR
for avg in avgs:
	#if avg.startswith('Raiza'): continue;
	plt.plot(avgs[avg]['BPP'], avgs[avg]['PSNR'], marker='.', color=avgs[avg]['Color'], 
			ls=avgs[avg]['Style'], label=avg, linewidth=2)
plt.xlabel('Bitrate (bpp)')
plt.ylabel('WS-PSNR (dB)')
plt.xlim(0, max_bpp)
plt.ylim(min_psnr, max_psnr)
plt.legend(frameon=False, bbox_to_anchor=(0.49, 0.5))
plt.savefig('bastos_WS-PSNR_' + target_file.split(".")[0] + '.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

# Plot WS-SSIM
for avg in avgs:
	#if avg.startswith('Raiza'): continue;
	plt.plot(avgs[avg]['BPP'], avgs[avg]['SSIM'], marker='.', color=avgs[avg]['Color'], 
			ls=avgs[avg]['Style'], label=avg, linewidth=2)
plt.xlabel('Bitrate (bpp)')
plt.ylabel('WS-SSIM (dB)')
plt.xlim(0, max_bpp)
plt.ylim(min_ssim, 1)
plt.legend(frameon=False, bbox_to_anchor=(0.49, 0.5))
plt.savefig('bastos_WS-SSIM_'  + target_file.split(".")[0] + '.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


