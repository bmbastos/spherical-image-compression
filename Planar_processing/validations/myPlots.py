import os
from scipy import signal
from pdb import set_trace as pause
from matplotlib import pyplot as plot
from operator import itemgetter
import csv
from numpy import *
import random

def pre_prossesing(csv_file_name:str) -> tuple:
	data_set = []
	methods = []
	with open(csv_file_name, 'r') as file:
		reader = csv.DictReader(file)
		methods_set = []
		for row in reader:
			file_name, method, psnr, ssim, bpp = row.values()
			methods.append(method)
			data_set.append({'Filename': file_name, 'Method': method, 'PSNR': eval(psnr), 'SSIM': eval(ssim), 'BPP':eval(bpp)})
	methods = list(set(methods))
	return data_set, methods
			

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
		if method == 'DCT':
			color = 'black'
			style = 'solid'
		else:
			style = 'solid' if 'propose' in method else 'dashed'
			color = 'red' if 'Oliveira' in method else 'blue'
		avgs[f'{method}']= {
			'PSNR': array(avg_psnr) / n_images,
			'SSIM': array(avg_ssim) / n_images,
			'BPP': array(avg_bpp) / n_images,
			'Color': color,
			'Style': style
		}
	return avgs
	

# __MAIN__#
target_file = 'standards.csv'
dataset, methods = pre_prossesing(target_file)
methods = list(sort(methods))
index_of_dct = methods.index('DCT')
methods.insert(0, methods.pop(index_of_dct))
avgs = averages(dataset, methods)
fig = plot.figure(label=target_file)
ax1 = fig.add_axes([0.1, 0.1 , 0.4, 0.8])
ax2 = fig.add_axes([0.55, 0.1 , 0.4, 0.8])
for avg in avgs:
	ax1.plot(avgs[avg]['BPP'], avgs[avg]['PSNR'], marker='o', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
	ax1.grid(True)
	ax1.set_xlabel('BPP'); ax1.set_ylabel('PSNR')
	ax1.set_xlim(0, 6)
	ax1.set_ylim(5, 45)
	ax1.legend(methods)
	ax2.plot(avgs[avg]['BPP'], avgs[avg]['SSIM'], marker='o', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
	ax2.grid(True)
	ax2.set_xlabel('BPP'); ax2.set_ylabel('SSIM')
	ax2.set_xlim(0, 6)
	ax2.set_ylim(0.4, 1)
	ax2.legend(methods)
plot.show()
	

