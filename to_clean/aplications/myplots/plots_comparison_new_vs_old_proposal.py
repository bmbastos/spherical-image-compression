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


def averages(data_set: list, methods: list, old_proposal:bool = True) -> dict:
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

		if 'JPEG' in method:
			color, style = 'black', 'solid'
			if not old_proposal:
				style = 'dotted'
		elif 'OLIVEIRA' in method:
			color, style = 'red', 'solid'
			if not old_proposal:
				style = 'dotted'
		elif 'BRAHIMI' in method:
			color, style = 'green', 'solid'
			if not old_proposal:
				style = 'dotted'
		elif 'RAIZA' in method:
			color, style = 'magenta', 'solid'
			if not old_proposal:
				style = 'dotted'
		elif 'DE_SIMONE' in method:
			color, style = 'blue', 'solid'
			if not old_proposal:
				style = 'dotted'
			


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
target_file_old = "our_proposal_4K.csv"
target_file_new = "new_proposal_4K.csv"
path_old = "aplications/main/results/" + target_file_old
path_new = "aplications/main/results/" + target_file_new
destination = "aplications/myplots/results/"
dataset_old, methods_old = pre_processing(path_old)
dataset_new, methods_new = pre_processing(path_new)

methods_old.sort()
methods_new.sort()
avgs_old = averages(dataset_old, methods_old, True)
avgs_new = averages(dataset_new, methods_new, False)
for avg in avgs_old:
	print(avg)
	olds = 0
	for index, psnr in enumerate(avgs_old[avg]['PSNR']):
		if index == 0: print(f'QF0{(index+1)*5} - OLD: {psnr:.2f} dB - NEW: {avgs_new[avg]["PSNR"][index]:.2f} dB')
		else: print(f'QF{(index+1)*5} - OLD: {psnr:.2f} dB - NEW: {avgs_new[avg]["PSNR"][index]:.2f} dB')
		if psnr > avgs_new[avg]["PSNR"][index]:
			olds += 1
	print(f'Este metodo com Qhvs tem {19-olds}/19 PSNRs superiores.')
	print()

pause()
margin_int = 5
margin_float = 0.2
max_psnr = max(round(max(max(avgs_old[avg]['PSNR']) for avg in avgs_old)), round(max(max(avgs_new[avg]['PSNR']) for avg in avgs_new))) + margin_int
max_bpp = max(max(max(avgs_old[avg]['BPP']) for avg in avgs_old), max(max(avgs_new[avg]['BPP']) for avg in avgs_new)) + margin_float
min_psnr = min(round(min(min(avgs_old[avg]['PSNR']) for avg in avgs_old)), round(min(min(avgs_new[avg]['PSNR']) for avg in avgs_new))) - margin_int
min_ssim = min(min(min(avgs_old[avg]['SSIM']) for avg in avgs_old), min(min(avgs_new[avg]['SSIM']) for avg in avgs_new)) - 0.03

plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.rcParams['text.usetex'] = True
#plt.rcParams["figure.figsize"] = (3, 2)

# Plot WS-PSNR
for avg in avgs_old:
	if 'RAIZA' in avg: continue
	if 'OLIVEIRA' in avg: continue
	if 'BRAHIMI' in avg: continue
	if 'DE_SIMONE' in avg: continue
	if 'JPEG' in avg: continue
	plt.plot(avgs_old[avg]['BPP'], avgs_old[avg]['PSNR'], marker='.', color=avgs_old[avg]['Color'], 
			ls=avgs_old[avg]['Style'], label=avg, linewidth=1)
for avg in avgs_new:
	if 'RAIZA' in avg: continue
	if 'OLIVEIRA' in avg: continue
	if 'BRAHIMI' in avg: continue
	if 'DE_SIMONE' in avg: continue
	if 'JPEG' in avg: continue
	plt.plot(avgs_new[avg]['BPP'], avgs_new[avg]['PSNR'], marker='.', color=avgs_new[avg]['Color'], 
			ls=avgs_new[avg]['Style'], label=avg, linewidth=1)
plt.xlabel('Bitrate (bpp)')
plt.ylabel('WS-PSNR (dB)')
plt.xlim(0, max_bpp)
plt.ylim(min_psnr, max_psnr)
#plt.legend(frameon=False, ncols=1, loc='upper left')
#plt.savefig(destination + 'bastos_WS-PSNR_' + target_file.split(".")[0] + '.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
plt.clf()
plt.cla()


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.rcParams['text.usetex'] = True
#plt.rcParams["figure.figsize"] = (3, 2)
# Plot WS-SSIM
for avg in avgs_old:
	if 'RAIZA' in avg: continue
	if 'OLIVEIRA' in avg: continue
	if 'BRAHIMI' in avg: continue
	if 'DE_SIMONE' in avg: continue
	if 'JPEG' in avg: continue
	plt.plot(avgs_old[avg]['BPP'], avgs_old[avg]['SSIM'], marker='.', color=avgs_old[avg]['Color'], 
			ls=avgs_old[avg]['Style'], label=avg, linewidth=1)
for avg in avgs_new:
	if 'RAIZA' in avg: continue
	if 'OLIVEIRA' in avg: continue
	if 'BRAHIMI' in avg: continue
	if 'DE_SIMONE' in avg: continue
	if 'JPEG' in avg: continue
	plt.plot(avgs_new[avg]['BPP'], avgs_new[avg]['SSIM'], marker='.', color=avgs_new[avg]['Color'], 
			ls=avgs_new[avg]['Style'], label=avg, linewidth=1)
plt.xlabel('Bitrate (bpp)')
plt.ylabel('WS-SSIM')
plt.xlim(0, max_bpp)
plt.ylim(min_ssim, 1+0.03)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#plt.legend(frameon=False, ncols=1, bbox_to_anchor=(0.45, 0.4))
#plt.legend(frameon=False, ncols=1, loc='lower right')
#plt.savefig(destination + 'bastos_WS-SSIM_'  + target_file.split(".")[0] + '.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


