import os
from scipy import signal
from pdb import set_trace as pause
from matplotlib import pyplot as plot
from matplotlib import colors
from operator import itemgetter
import csv
from numpy import *
import random
import ast

def pre_prossesing(csv_file_name:str) -> tuple:
    data_set = []
    methods = []
    with open(csv_file_name, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_name, method, psnr, ssim, bpp = row.values()
            methods.append(method)
            # Converta os valores de PSNR, SSIM, e BPP para float
            data_set.append({
                'Filename': file_name,
                'Method': method,
                'PSNR': array(list(map(float, ast.literal_eval(psnr))), dtype=float),
                'SSIM': array(list(map(float, ast.literal_eval(ssim))), dtype=float),
                'BPP': array(list(map(float, ast.literal_eval(bpp))), dtype=float)
            })
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
				style = 'solid'
			if str.startswith(method, 'Brahimi'):
				color = 'green'
				style = 'solid'
			if str.startswith(method, 'Raiza'):
				color = 'blue'
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
target_file = 'Brahimi_FAdjusted_BAdjusted.csv'
dataset, methods = pre_prossesing(target_file)
methods = list(sort(methods))
index_of_dct = methods.index("JPEG Planar")
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
