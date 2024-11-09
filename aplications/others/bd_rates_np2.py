import csv
from numpy import array, zeros
import bjontegaard as bd
from pdb import set_trace as pause
from matplotlib import pyplot as plt

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
""" Realiza o pré processamento dos dados do arquivo CSV. """

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
			#method = r'JPEG [6]'
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
""" Calcula a média dos valores de PSNR, SSIM e BPP para cada método. """

def check_increase_order(ndarray:array) -> bool:
	output = True
	current = ndarray[0]
	for value in ndarray[1:]:
		if value < current:
			return False
		current = value
	return output

target_file = 'results/test_np2_C_transformation.csv'
data_set, methods = pre_processing(target_file)
methods.sort()
methods = [m for m in methods if m.startswith('JPEG')] + [m for m in methods if not m.startswith('JPEG')]
avgs = averages(data_set, methods)

rate_anchor = avgs[methods[0]]['BPP']
if not check_increase_order(rate_anchor): exit('Error: BPP anchor values are not in increasing order')
psnr_anchor = avgs[methods[0]]['PSNR']
if not check_increase_order(psnr_anchor): exit('Error: PSNR anchor values are not in increasing order')
for avg in avgs:
	if avg != methods[0]:
		print(avg)
		rate_teste = avgs[avg]['BPP']
		if not check_increase_order(rate_teste): exit('Error: BPP test values are not in increasing order')
		psnr_teste = avgs[avg]['PSNR']
		print(psnr_teste)
		if not check_increase_order(psnr_teste): exit('Error: PSNR test values are not in increasing order')
		
		print(bd.bd_rate(rate_anchor, psnr_anchor, rate_teste, psnr_teste, method='akima'))
		print(bd.bd_rate(rate_teste, psnr_teste, rate_anchor, psnr_anchor, method='akima'))
		pause()
