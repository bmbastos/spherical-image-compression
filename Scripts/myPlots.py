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
				color = 'orange'
				style = 'solid'
			if str.startswith(method, 'Raiza'):
				color = 'm'
				style = 'solid'
			else:
				if len(methods) == 3:
					if 'ZR' in method:
						color = 'red'
						style = 'solid'
					if 'ZO' in method:
						color = 'blue'
						style = 'solid'
				else:
					if str.endswith(method, "})$"):
						color = 'red'
						style = 'solid'
					if str.endswith(method, "phi)$"):
						color = 'green'
						style = 'dashed'
					if str.endswith(method, "phi$"):
						color = 'blue'
						style = 'solid'

			"""
			if method == r"$\operatorname{np2}((\mathbf{Q})_\phi\bigstar\mathbf{ZR})$" or method == r"$\operatorname{np2}((\mathbf{Q})_\phi\bigstar\mathbf{ZO})$":
				color = 'red'
				style = 'solid'
			if method == r"$\operatorname{np2}((\mathbf{Q}\bigstar\mathbf{ZR})_\phi)$" or method == r"$\operatorname{np2}((\mathbf{Q}\bigstar\mathbf{ZO})_\phi)$":
				color = 'green'
				style = 'dashed'
			if method == r"$(\operatorname{np2}(\mathbf{Q})\bigstar\mathbf{ZR})_\phi$" or method == r"$(\operatorname{np2}(\mathbf{Q})\bigstar\mathbf{ZO})_\phi$":
				color = 'blue'
				style = 'solid'
			"""
				
		avgs[f'{method}']= {
			'PSNR': array(avg_psnr) / n_images,
			'SSIM': array(avg_ssim) / n_images,
			'BPP': array(avg_bpp) / n_images,
			'Color': color,
			'Style': style
		}
	return avgs
	

# __MAIN__#
target_file = 'JPEG_RDCTs_SphericalMethods_in_SphericalImages_with_np2.csv'
dataset, methods = pre_prossesing(target_file)
methods = list(sort(methods))
index_of_dct = methods.index("JPEG Spherical")
methods.insert(0, methods.pop(index_of_dct))
avgs = averages(dataset, methods)
fig = plot.figure(label=target_file)
ax1 = fig.add_axes([0.1, 0.1 , 0.4, 0.8])
ax2 = fig.add_axes([0.55, 0.1 , 0.4, 0.8])
for avg in avgs:
	ax1.plot(avgs[avg]['BPP'], avgs[avg]['PSNR'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
	ax1.grid(True)
	ax1.set_xlabel('BPP'); ax1.set_ylabel('PSNR')
	ax1.set_xlim(0, 3)
	ax1.set_ylim(25, 50)
	ax1.legend(methods)  # Não é necessário usar r"{}".format()

	ax2.plot(avgs[avg]['BPP'], avgs[avg]['SSIM'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
	ax2.grid(True)
	ax2.set_xlabel('BPP'); ax2.set_ylabel('SSIM')
	ax2.set_xlim(0, 3)
	ax2.set_ylim(0.7, 1)
	ax2.legend(methods)  # Não é necessário usar r"{}".format()
plot.show()
