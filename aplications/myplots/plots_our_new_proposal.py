import os
import sys
import random
import csv
from numpy import array, zeros, ceil
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pdb import set_trace as pause

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
		label = ''

		if method.startswith('JPEG'):
			color, style = 'black', 'solid'
			label = r'$\mathbf{C}, \mathbf{Q}_{(QF)}$'
		elif method.startswith('OUR_P2'):
			color, style = 'orange', 'solid'
			label = r'$\mathbf{T}_3, \mathbf{\widetilde{Q}}_{\phi,(QF)}*$'
		elif method.startswith('OUR_P3'):
			color, style = 'green', 'dotted'
			label = r'$\mathbf{T}_3, \mathbf{\widetilde{Q}}_{\phi,(QF)}$'
		elif method.startswith('OUR_P4'):
			color, style = 'red', 'solid'
			label = r'$\mathbf{T}_3, \mathbf{\widetilde{Q}}_{\phi,(QF)}$'
		elif method.startswith('DE_SIMONE'):
			color, style = 'blue', 'solid'
			label = r'$\mathbf{C}, \mathbf{Q}_{\phi,(QF)}$'


		avgs[method] = {
			'PSNR': avg_psnr / n_images,
			'SSIM': avg_ssim / n_images,
			'BPP': avg_bpp / n_images,
			'Label': label,
			'Color': color,
			'Style': style
		}
	return avgs

# __MAIN__#
print("Current path:", os.getcwd())
target_file ="our_new_proposal_4K.csv"
path = "aplications/main/results/" + target_file
destination = "aplications/myplots/results/"
dataset, methods = pre_processing(path)
avgs = averages(dataset, methods)
sorted_avgs = sorted(avgs.values(), key=lambda x: x['Label'])

margin_int = 5
margin_float = 0.2
max_psnr = round(max(max(avgs[avg]['PSNR']) for avg in avgs)) + margin_int
max_bpp = max(max(avgs[avg]['BPP']) for avg in avgs) + margin_float
min_psnr = round(min(min(avgs[avg]['PSNR']) for avg in avgs)) - margin_int
min_ssim = min(min(avgs[avg]['SSIM']) for avg in avgs) - 0.03

plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (3, 2)

# Plot WS-PSNR
for avg in sorted_avgs:
	plt.plot(avg['BPP'], avg['PSNR'], marker='.', color=avg['Color'], 
			ls=avg['Style'], label=avg['Label'], linewidth=1)
plt.xlabel('Bitrate (bpp)')
plt.ylabel('WS-PSNR (dB)')
plt.xlim(0, max_bpp)
plt.ylim(25, max_psnr+10)
plt.legend(frameon=False, ncols=2, loc='upper left')
plt.savefig(destination + 'bastos_Ws-PSNR_' + target_file.split(".")[0] + '.pdf', bbox_inches='tight', pad_inches=0)
#plt.show()
plt.clf()
plt.cla()


# Plot WS-SSIM
for avg in sorted_avgs:
	plt.plot(avg['BPP'], avg['SSIM'], marker='.', color=avg['Color'], 
			ls=avg['Style'], label=avg['Label'], linewidth=1)
plt.xlabel('Bitrate (bpp)')
plt.ylabel('WS-SSIM')
plt.xlim(0, max_bpp)
plt.ylim(0.7, 1+0.01)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#plt.legend(frameon=False, ncols=1, bbox_to_anchor=(0.45, 0.4))
plt.legend(frameon=False, ncols=1, loc='lower right', markerfirst=False)
plt.savefig(destination + 'bastos_WS-SSIM_'  + target_file.split(".")[0] + '.pdf', bbox_inches='tight', pad_inches=0)
#plt.show()

