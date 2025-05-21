import csv
from numpy import array, zeros, ceil
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pdb import set_trace as pause
import pandas as pd

data_file = pd.read_csv('./outputs/csv_files/benchmark_results.csv')
methods = data_file['Method'].unique()
qfs = data_file['Quality Factor'].unique()
means_psnr = data_file.groupby(['Method', 'Quality Factor'])['WS-PSNR'].mean().reset_index()
means_ssim = data_file.groupby(['Method', 'Quality Factor'])['WS-SSIM'].mean().reset_index()
means_bpp = data_file.groupby(['Method', 'Quality Factor'])['BPP'].mean().reset_index()

for method in methods:
	psnr_values = means_psnr[means_psnr['Method'] == method]['WS-PSNR'].values
	#ssim_values = means_ssim[means_ssim['Method'] == method]['WS-SSIM'].values
	bpp_values = means_bpp[means_bpp['Method'] == method]['BPP'].values

	# Plot PSNR
	plt.plot(bpp_values, psnr_values, label=method, marker='.')
plt.xlabel('BPP')
plt.ylabel('PSNR (dB)')
plt.title('Average PSNR vs BPP')
plt.legend()
plt.grid()
plt.show()

plt.clf()
plt.cla()

for method in methods:
	ssim_values = means_ssim[means_ssim['Method'] == method]['WS-SSIM'].values
	pause()
	bpp_values = means_bpp[means_bpp['Method'] == method]['BPP'].values

	# Plot SSIM
	plt.plot(bpp_values, ssim_values, label=method, marker='x')
plt.xlabel('BPP')
plt.ylabel('SSIM')
plt.title('Average SSIM vs BPP')
plt.legend()
plt.grid()
plt.show()





		
