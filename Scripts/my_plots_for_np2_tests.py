import os
from scipy import signal
from pdb import set_trace as pause
from matplotlib import pyplot as plot
from matplotlib import colors
from operator import itemgetter
import csv
from numpy import *
import random

def pre_prossesing(csv_file_name:str) -> tuple:
	data_set = []
	methods = []
	with open(csv_file_name, 'r') as file:
		reader = csv.DictReader(file)
		for row in reader:
			file_name, method, psnr, ssim, bpp = row.values()
			methods.append(method)
			data_set.append({'Filename': file_name, 'Method': method, 'PSNR': eval(psnr), 'SSIM': eval(ssim), 'BPP':eval(bpp)})
	methods = list(set(methods))
	return data_set, methods

def random_unique_colors(num_cores):
	colors_css4 = list(colors.CSS4_COLORS.values())
	dark_colors = [color for color in colors_css4 if colors.to_rgb(color)[2] < 0.5] # Seleciona apenas cores com valor de intensidade menor que 0.5
	unique_colors = set()
	while len(unique_colors) < num_cores:
		random_color = random.choice(dark_colors)
		unique_colors.add(random_color)
	return list(unique_colors)
			

def averages(data_set: list, methods: list) -> dict:
	n_images = int(len(data_set) / len(methods))
	avgs = {}
	print(f'Quantidade de imagens {n_images}')
	for method in methods:
		avg_psnr = zeros(19, dtype=float)
		avg_ssim = zeros(19, dtype=float)
		avg_bpp = zeros(19, dtype=float)
		for data in data_set:
			if data['Method'] == method:
				avg_psnr = list(map(sum, zip(avg_psnr, data['PSNR'])))
				avg_ssim = list(map(sum, zip(avg_ssim, data['SSIM'])))
				avg_bpp = list(map(sum, zip(avg_bpp, data['BPP'])))
		style = 'solid'
		if str.startswith(method, 'JPEG'):
			color = 'black'
		else:
			color = 'purple'
			style = 'solid'
			if str.startswith(method, 'Oliveira'):
				color = 'red'
			elif str.startswith(method, 'Raiza'):
				color = 'blue'
			if str.endswith(method, 'R') or str.endswith(method, 'Spherical'):
				style = 'dashed'
			elif str.endswith(method, 'C') or str.endswith(method, 'Planar'):
				style = 'solid'
			elif str.endswith(method, 'F'):
				style = 'dotted'	
				
		avgs[f'{method}']= {
			'PSNR': array(avg_psnr) / n_images,
			'SSIM': array(avg_ssim) / n_images,
			'BPP': array(avg_bpp) / n_images,
			'Color': color,
			'Style': style
		}
	return avgs
	

# __MAIN__#
target_file = 'Teste_de_np2_rounded_forward_with_LowCost.csv'
dataset, methods = pre_prossesing(target_file)
methods = list(sort(methods))
index_of_dct = methods.index("JPEG Spherical")
methods.insert(0, methods.pop(index_of_dct))
avgs = averages(dataset, methods)

(fig, axes) = plot.subplots(3, 2, figsize=(10, 10), label=target_file)

for avg in avgs:
	if 'JPEG' in avg or 'RC' in avg or 'CC' in avg or 'FC' in avg:
		axes[0,0].plot(avgs[avg]['BPP'], avgs[avg]['PSNR'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
		axes[0,0].grid(True)
		axes[0,0].set_xlabel('BPP')
		axes[0,0].set_ylabel('PSNR')
		axes[0,0].set_xlim(0, 3)
		axes[0,0].set_ylim(0.7, 50)
		axes[0,0].legend()

		axes[0,1].plot(avgs[avg]['BPP'], avgs[avg]['SSIM'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
		axes[0,1].grid(True)
		axes[0,1].set_xlabel('BPP')
		axes[0,1].set_ylabel('SSIM')
		axes[0,1].set_xlim(0, 3)
		axes[0,1].set_ylim(0.5, 1)
		axes[0,1].legend()
	
	if 'JPEG' in avg or 'RR' in avg or 'CR' in avg or 'FR' in avg:
		axes[1,0].plot(avgs[avg]['BPP'], avgs[avg]['PSNR'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
		axes[1,0].grid(True)
		axes[1,0].set_xlabel('BPP')
		axes[1,0].set_ylabel('PSNR')
		axes[1,0].set_xlim(0, 3)
		axes[1,0].set_ylim(0.7, 50)
		axes[1,0].legend()

		axes[1,1].plot(avgs[avg]['BPP'], avgs[avg]['SSIM'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
		axes[1,1].grid(True)
		axes[1,1].set_xlabel('BPP')
		axes[1,1].set_ylabel('SSIM')
		axes[1,1].set_xlim(0, 3)
		axes[1,1].set_ylim(0.5, 1)
		axes[1,1].legend()

	if 'JPEG' in avg or 'RF' in avg or 'CF' in avg or 'FF' in avg:
		axes[2,0].plot(avgs[avg]['BPP'], avgs[avg]['PSNR'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
		axes[2,0].grid(True)
		axes[2,0].set_xlabel('BPP')
		axes[2,0].set_ylabel('PSNR')
		axes[2,0].set_xlim(0, 3)
		axes[2,0].set_ylim(0.7, 50)
		axes[2,0].legend()
	
		axes[2,1].plot(avgs[avg]['BPP'], avgs[avg]['SSIM'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
		axes[2,1].grid(True)
		axes[2,1].set_xlabel('BPP')
		axes[2,1].set_ylabel('SSIM')
		axes[2,1].set_xlim(0, 3)
		axes[2,1].set_ylim(0.5, 1)
		axes[2,1].legend()
plot.tight_layout()
plot.show()
