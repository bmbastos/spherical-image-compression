import os
from scipy import signal
from pdb import set_trace as pause
from matplotlib import pyplot as plt
from matplotlib import colors
from operator import itemgetter
import csv
from numpy import *
import random

def pre_processing(csv_file_name: str) -> tuple:
	data_set = []
	methods = []
	with open(csv_file_name, 'r') as file:
		reader = csv.DictReader(file)
		for row in reader:
			file_name, method, psnr, ssim, bpp = row.values()
			methods.append(method)

			# Helper function to clean and convert values
			def clean_and_convert(value_str):
				return list(map(float, value_str.replace('np.float64(', '').replace(')', '').strip('[]').split(',')))

			data_set.append({
				'Filename': file_name,
				'Method': method,
				'PSNR': clean_and_convert(psnr),
				'SSIM': clean_and_convert(ssim),
				'BPP': clean_and_convert(bpp)
			})
	
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
		style = 'solid'
		color = 'purple'
		if 'JPEG' in method:
			color = 'black'
		elif 'Brahimi' in method:
			color = 'green'
		elif 'HVS' in method:
			color = 'gray'
		if str.endswith(method, 'R'):
			style = 'dashed'
		elif str.endswith(method, 'C'):
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
target_file = 'aplications/others/results/teste_np2_R_forward.csv'
destination = "aplications/myplots/results/np2_arrangements/"
dataset, methods = pre_processing(target_file)
methods = list(sort(methods))
avgs = averages(dataset, methods)

int_margin = 5
float_margin = 0.05
max_psnr = 50 + int_margin
min_psnr = 10 - int_margin
max_ssim = 1 + float_margin
min_ssim = 0.5 - float_margin
max_bpp = 3.95 + float_margin
min_bpp = 0.0

prefix = ''
sufixes = ['C', 'F', 'R']
if 'np2_C_forward' in target_file:
	prefix = 'C'
elif 'np2_F_forward' in target_file:
	prefix = 'F'
elif 'np2_R_forward' in target_file:
	prefix = 'R'

for sufix in sufixes:
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = 8
	plt.rcParams['text.usetex'] = True
	plt.rcParams["figure.figsize"] = (3, 2)
	# (PSNR)
	for avg in avgs:
		if (prefix + sufix) in avg:
			plt.plot(avgs[avg]['BPP'], avgs[avg]['PSNR'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
			plt.xlabel('Bitrate (bpp)')
			plt.ylabel('WS-PSNR (dB)')
			plt.xlim(0, max_bpp)
			plt.ylim(min_psnr, max_psnr)
			plt.legend(frameon=False, ncols=1)
	plt.savefig(destination + 'arragement_np2_' + prefix + sufix + '_PSNR.pdf', bbox_inches='tight', pad_inches=0)
	plt.clf()
	plt.cla()

	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = 8
	plt.rcParams['text.usetex'] = True
	plt.rcParams["figure.figsize"] = (3, 2)
	# (SSIM)
	for avg in avgs:
		if (prefix + sufix) in avg:
			plt.plot(avgs[avg]['BPP'], avgs[avg]['SSIM'], marker='.', color = avgs[avg]['Color'], ls=avgs[avg]['Style'], label=avg)
			plt.xlabel('Bitrate (bpp)')
			plt.ylabel('WS-SSIM')
			plt.xlim(0, max_bpp)
			plt.ylim(min_ssim, max_ssim)
			plt.legend(frameon=False, ncols=1)
	plt.savefig(destination + 'arragement_np2_' + prefix + sufix + '_SSIM.pdf', bbox_inches='tight', pad_inches=0)
	plt.clf()
	plt.cla()













