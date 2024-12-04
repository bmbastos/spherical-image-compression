import csv
from numpy import array, zeros
import bjontegaard as bd
from pdb import set_trace as pause
from matplotlib import pyplot as plt

def clean_and_convert(value_str:str):
	return list(map(float, value_str.replace('np.float64(', '').replace(')', '').strip('[]').split(',')))

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
				'PSNR': clean_and_convert(psnr),
				'SSIM': clean_and_convert(ssim),
				'BPP': clean_and_convert(bpp)
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
		elif method.startswith('BRAHIMI'):
			color, style = 'green', 'dotted'
		elif method.startswith('RAIZA'):
			color, style = 'magenta', 'dotted'
		elif method.startswith('DE_SIMONE'):
			color, style = 'blue', 'dotted'


		avgs[method] = {
			'PSNR': avg_psnr / n_images,
			'SSIM': avg_ssim / n_images,
			'BPP': avg_bpp / n_images,
			'Color': color,
			'Style': style
		}
	return avgs
""" Calcula a média dos valores de PSNR, SSIM e BPP para cada método. """


def filter_data_by_bpp(data_set: list, threshold:float = 0.5) -> tuple:
	filtered_data = {}
	quantidade = 100
	for data in data_set:
		psnr = [float(data_set[data]['PSNR'][0])]
		ssim = [float(data_set[data]['SSIM'][0])]
		bpp = [float(data_set[data]['BPP'][0])]
		current_psnr = float(data_set[data]['PSNR'][0])
		current_ssim = float(data_set[data]['SSIM'][0])
		for index, bitrate in enumerate(data_set[data]['BPP'], start=1):
			passou_psnr = False
			passou_ssim = False
			if bitrate <= threshold:
				if data_set[data]['PSNR'][index] > current_psnr:
					current_psnr = data_set[data]['PSNR'][index]
					psnr.append(float(data_set[data]['PSNR'][index]))
					passou_psnr = True
				if data_set[data]['SSIM'][index] > current_ssim:
					current_ssim = data_set[data]['SSIM'][index]
					ssim.append(float(data_set[data]['SSIM'][index]))
					passou_ssim = True
				if passou_psnr and passou_ssim:
					bpp.append(float(data_set[data]['BPP'][index]))
		tam_psnr = len(psnr)
		#print(f"{data}: PSNR = {tam_psnr}")
		tam_ssim = len(ssim)
		#print(f"{data}: SSIM = {tam_ssim}")
		if tam_psnr < quantidade:
			quantidade = tam_psnr
		if tam_ssim < quantidade:
			quantidade = tam_ssim
		filtered_data[data] = {'PSNR': psnr, 'SSIM':ssim, 'BPP': bpp}
	#print(f"Quantidade = {quantidade}")
	return filtered_data, quantidade

def find_best_points(filtered_data: dict, quantidade: int) -> dict:
    best_points = {}
    for data in filtered_data:
        best_points[data] = {'PSNR': [], 'SSIM': [], 'BPP': []}
        for metric in ['PSNR', 'SSIM', 'BPP']:
            values = filtered_data[data][metric]
            step = max(1, len(values) // quantidade)
            best_points[data][metric] = [values[i * step] for i in range(min(quantidade, len(values) // step))]
    return best_points



target_file ="our_proposal_4K.csv"
path = "aplications/main/results/" + target_file
destination = "aplications/main/results/"
data_set, methods = pre_processing(path)
methods.sort()
methods = [m for m in methods if m.startswith('JPEG')] + [m for m in methods if not m.startswith('JPEG')]
avgs = averages(data_set, methods)
with open(destination+'averages'+target_file, 'w', newline='') as csvfile:
	fieldnames = ['Method', 'PSNR', 'SSIM', 'BPP']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

	writer.writeheader()
	for method, data in avgs.items():
		writer.writerow({
			'Method': method,
			'PSNR': ','.join(map(str, data['PSNR'])),
			'SSIM': ','.join(map(str, data['SSIM'])),
			'BPP': ','.join(map(str, data['BPP']))
		})

dados, quantidade = filter_data_by_bpp(avgs)
best_datas = find_best_points(dados, quantidade)
for data in best_datas:
	print(data)
	print(best_datas[data]['PSNR'])
	print(len(best_datas[data]['PSNR']))
	print(best_datas[data]['SSIM'])
	print(len(best_datas[data]['SSIM']))
	print(best_datas[data]['BPP'])
	print(len(best_datas[data]['BPP']))
	print()


for data in best_datas:
	rate_anchor = best_datas[data]['BPP']
	psnr_anchor = best_datas[data]['PSNR']
	for data_test in best_datas:
		if data != data_test:
			rate_teste = best_datas[data_test]['BPP']
			psnr_teste = best_datas[data_test]['PSNR']
			print(f"{data} vs {data_test}")
			#plt.plot(rate_anchor, psnr_anchor, label=data)
			#plt.plot(rate_teste, psnr_teste, label=data_test)
			#plt.legend()
			print('BD-Rate: ', end='')
			print(bd.bd_rate(rate_anchor, psnr_anchor, rate_teste, psnr_teste, method='akima', min_overlap=0))
			print('BD-PSNR: ', end='')
			print(bd.bd_psnr(rate_anchor, psnr_anchor, rate_teste, psnr_teste, method='akima', min_overlap=0))
			print()
			#plt.show()
			
"""
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
"""