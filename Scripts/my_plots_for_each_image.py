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
	filenames = []
	with open(csv_file_name, 'r') as file:
		reader = csv.DictReader(file)
		for row in reader:
			file_name, method, psnr, ssim, bpp = row.values()
			methods.append(method)
			filenames.append(file_name)
			data_set.append({'Filename': file_name, 'Method': method, 'PSNR': eval(psnr), 'SSIM': eval(ssim), 'BPP':eval(bpp)})
	methods = list(set(methods))
	filenames = list(set(filenames))
	return data_set, methods, filenames

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
		elif str.startswith(method, 'Oliveira'):
			color = 'red'
		elif str.startswith(method, 'Brahimi'):
			color = 'green'
		elif str.startswith(method, 'Raiza'):
			color = 'blue'
		if 'Spherical' in method:
			style = 'dashed'
		elif 'Planar' in method:
			style = 'solid'
			
				
		avgs[f'{method}']= {
			'PSNR': array(avg_psnr) / n_images,
			'SSIM': array(avg_ssim) / n_images,
			'BPP': array(avg_bpp) / n_images,
			'Color': color,
			'Style': style
		}
	return avgs
	

# __MAIN__#
target_file = 'Teste_Raiza_Oliveira_Spherical_X_Planar.csv'
dataset, methods, filenames = pre_prossesing(target_file)
for filename in filenames:
	(fig, axes) = plot.subplots(2, 2, figsize=(10, 10), label=filename)	
	for element in dataset:
		if element['Filename'] == filename:
			if 'JPEG' in element['Method']:
				color = 'black'
				if 'Spherical' in element['Method']:
					style = 'dashed'
				else:
					style = 'solid'
			elif 'Oliveira' in element['Method']:
				color = 'red'
				if 'Spherical' in element['Method']:
					style = 'dashed'
				else:
					style = 'solid'
			elif 'Brahimi' in element['Method']:
				color = 'green'
				if 'Spherical' in element['Method']:
					style = 'dashed'
				else:
					style = 'solid'
			elif 'Raiza' in element['Method']:
				color = 'blue'
				if 'Spherical' in element['Method']:
					style = 'dashed'
				else:
					style = 'solid'
			axes[0,0].plot(element['BPP'], element['PSNR'], marker='.', color = color, ls=style, label=element['Method'])
			axes[0,1].plot(element['BPP'], element['SSIM'], marker='.', color = color, ls=style, label=element['Method'])
			axes[1,0].plot(element['BPP'], element['PSNR'], marker='.', color = color, ls=style, label=element['Method'])
			axes[1,1].plot(element['BPP'], element['SSIM'], marker='.', color = color, ls=style, label=element['Method'])
			for i in range(0,2):
				for j in range(0,2):
					axes[i,j].grid(True)
					axes[i,j].set_xlabel('BPP')
					axes[i,j].legend()
					if j == 0:
						axes[i,j].set_ylabel('PSNR')
						axes[i,j].set_xlim(0, 3)
						axes[i,j].set_ylim(25, 50)
					else:
						axes[i,j].set_ylabel('SSIM')
						axes[i,j].set_xlim(0, 3)
						axes[i,j].set_ylim(0.7, 1)

plot.show()

"""

methods = list(sort(methods))
index_of_dct = methods.index("JPEG Spherical")
methods.insert(0, methods.pop(index_of_dct))
avgs = averages(dataset, methods)
max_psnr = round(max([max(avgs[avg]['PSNR']) for avg in avgs])) + 3
max_bpp = ceil(max([max(avgs[avg]['BPP']) for avg in avgs]))
min_psnr = round(min([min(avgs[avg]['PSNR']) for avg in avgs])) - 3
min_ssim = min([min(avgs[avg]['SSIM']) for avg in avgs]) - 0.03
fig = plot.figure(label=target_file)
ax1 = fig.add_axes([0.1, 0.1 , 0.4, 0.8])
ax2 = fig.add_axes([0.55, 0.1 , 0.4, 0.8])
for avg in avgs:
	ax1.plot(avgs[avg]['BPP'], avgs[avg]['PSNR'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
	ax1.grid(True)
	ax1.set_xlabel('BPP'); ax1.set_ylabel('PSNR')
	ax1.set_xlim(0, max_bpp)
	ax1.set_ylim(min_psnr, max_psnr)
	ax1.legend(methods)

	ax2.plot(avgs[avg]['BPP'], avgs[avg]['SSIM'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
	ax2.grid(True)
	ax2.set_xlabel('BPP'); ax2.set_ylabel('SSIM')
	ax2.set_xlim(0, max_bpp)
	ax2.set_ylim(min_ssim, 1)
	ax2.legend(methods)
plot.show()
"""