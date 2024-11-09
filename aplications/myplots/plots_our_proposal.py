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
		elif method.startswith('OLIVEIRA'):
			color, style = 'red', 'dotted'
			method = 'Proposed II'
		elif method.startswith('BRAHIMI'):
			color, style = 'green', 'dotted'
			method = 'Proposed III'
		elif method.startswith('RAIZA'):
			color, style = 'magenta', 'dotted'
			method = 'Proposed I'
		elif method.startswith('DE_SIMONE'):
			color, style = 'blue', 'solid'
			method = r'De Simone et al. [6]'


		avgs[method] = {
			'PSNR': avg_psnr / n_images,
			'SSIM': avg_ssim / n_images,
			'BPP': avg_bpp / n_images,
			'Color': color,
			'Style': style
		}
	return avgs

# __MAIN__#
print("Current path:", os.getcwd())
target_file ="our_proposal.csv"
path = "aplications/main/results/" + target_file
destination = "aplications/myplots/results/"
dataset, methods = pre_processing(path)
methods.sort()
methods.insert(0, methods.pop(methods.index("DE_SIMONE")))
avgs = averages(dataset, methods)

max_psnr = round(max(max(avgs[avg]['PSNR']) for avg in avgs)) + 5
#max_bpp = ceil(max(max(avgs[avg]['BPP']) for avg in avgs))
min_psnr = round(min(min(avgs[avg]['PSNR']) for avg in avgs)) - 3
min_ssim = min(min(avgs[avg]['SSIM']) for avg in avgs) - 0.03

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (3, 2)

# Plot WS-PSNR
for avg in avgs:
	if avg.startswith('JPEG'): continue;
	plt.plot(avgs[avg]['BPP'], avgs[avg]['PSNR'], marker='.', color=avgs[avg]['Color'], 
			ls=avgs[avg]['Style'], label=avg, linewidth=1)
plt.xlabel('Bitrate (bpp)')
plt.ylabel('WS-PSNR (dB)')
plt.xlim(0, 2.5)
plt.ylim(min_psnr, max_psnr+5)
plt.legend(frameon=False, ncols=1, loc='upper left')
plt.savefig(destination + '2bastos_WS-PSNR_' + target_file.split(".")[0] + '.pdf', bbox_inches='tight', pad_inches=0)
#plt.show()
plt.clf()
plt.cla()

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (3, 2)
# Plot WS-SSIM
for avg in avgs:
	if avg.startswith('JPEG'): continue;
	plt.plot(avgs[avg]['BPP'], avgs[avg]['SSIM'], marker='.', color=avgs[avg]['Color'], 
			ls=avgs[avg]['Style'], label=avg, linewidth=1)
plt.xlabel('Bitrate (bpp)')
plt.ylabel('WS-SSIM')
plt.xlim(0, 2.5)
plt.ylim(min_ssim-0.05, 1+0.03)
#plt.legend(frameon=False, ncols=1, bbox_to_anchor=(0.45, 0.4))
plt.legend(frameon=False, ncols=1, loc='lower right')
plt.savefig(destination + '2bastos_WS-SSIM_'  + target_file.split(".")[0] + '.pdf', bbox_inches='tight', pad_inches=0)
#plt.show()


