import csv
from numpy import array, zeros
from operator import itemgetter
from pdb import set_trace as pause


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

def averages(data_set: list, methods: list) -> dict:
	n_images = len(data_set) // len(methods)
	avgs = {}
	for method in methods:
		avg_psnr = zeros(19, dtype=float)
		avg_ssim = zeros(19, dtype=float)
		avg_bpp = zeros(19, dtype=float)
		for data in data_set:
			if data['Method'] == method:
				avg_psnr += array(data['PSNR'])
				avg_ssim += array(data['SSIM'])
				avg_bpp += array(data['BPP'])
		avgs[method] = {
			'PSNR': avg_psnr / n_images,
			'SSIM': avg_ssim / n_images,
			'BPP': avg_bpp / n_images
		}
	return avgs

# __MAIN__#
target_file = input('Digite o nome do arquivo: ') + '.csv'
data_set, methods = pre_processing('our_proposal.csv')
avgs = averages(data_set, methods)

fieldnames = ['Method', 'PSNR', 'SSIM', 'BPP']
with open('avgs_' + target_file, 'w') as csv_file:
	writer = csv.DictWriter(csv_file, fieldnames)
	writer.writeheader()
	for method in avgs:
		writer.writerow({
			'Method': method,
			'PSNR': avgs[method]['PSNR'].tolist(),
			'SSIM': avgs[method]['SSIM'].tolist(),
			'BPP': avgs[method]['BPP'].tolist()
		})
