import os
from matrixes import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
from scipy import signal
from pdb import set_trace as pause
from matplotlib import pyplot as plot
from time import time
from tools import *
from datetime import datetime
from tqdm import tqdm
from operator import itemgetter
import csv

#  Definição das funções auxiliares 
def quantize(quality_factor:int, quantization_matrix:ndarray) -> ndarray:
	s = 0.0
	if quality_factor < 50:
		s = 5_000 / quality_factor
	else:
		s = 200 - (2 * quality_factor)
	resulting_matrix = floor((s * quantization_matrix + 50) / 100)
	return resulting_matrix
""" Calcula a matriz de quantização dado um fator de quantização """

def calculate_matrix_of_transformation(k:int) -> ndarray:
	row = 0
	alpha = 0.0
	transformation_matrix = zeros((k, k), float64)
	while(row < k):								
		if row == 0:
			alpha = 1 / (k ** 0.5)
		else:
			alpha = (2 / k) ** 0.5
		col = 0
		while(col < k):
			transformation_matrix[row][col] = alpha * cos((pi * row * (2 * col + 1)) / (2 * k))
			col += 1
		row += 1
	return transformation_matrix
""" Calcula a matriz de transformação com base no tamanho do bloco """

def bpp(quantized_image:ndarray):
	return count_nonzero(logical_not(isclose(quantized_image, 0))) * 8 / (quantized_image.shape[0]*quantized_image.shape[1])

def compute_scale_matrix(transformation_matrix:ndarray) -> matrix:
	scale_matrix = matrix(sqrt(linalg.inv(dot(transformation_matrix, transformation_matrix.T))))
	scale_vector = matrix(diag(scale_matrix))
	return scale_matrix, scale_vector
""" Matrix diagonal e elementos da matriz diagonal vetorizados """

def np2_round(quantization_matrix:matrix) -> matrix:
	return power(2, around(log2(quantization_matrix)))
""" Função que calcula as potencias de dois mais próximas de uma dada matriz - Oliveira """

def np2_ceil(quantization_matrix:matrix) -> matrix:
	return power(2, ceil(log2(quantization_matrix)))
"""Função de transformação de uma matriz em uma matriz de potências de dois - Brahimi """


# Estrturas de armazenamento e constantes de pré-processamento
quality_factors = range(5, 100, 5)
METHODS = {
    'DCT': {'Label': 'DCT', 'Color': 'black', 'Style': 'solid', 'Legend': 'DCT'},
    'Oliveira-propose': {'Label': 'Oliveira-propose', 'Color': 'blue', 'Style': 'dashed', 'Legend': 'Oliveira',},
    'Brahimi-propose': {'Label': 'Brahimi-propose', 'Color': 'green', 'Style': 'dashed', 'Legend': 'Brahimi'}

}
DATAS = {method: {'PSNR': zeros(len(quality_factors)), 'SSIM': zeros(len(quality_factors)), 'BPP': zeros(len(quality_factors))} for method in METHODS}
T = calculate_matrix_of_transformation(8)
SO, so = compute_scale_matrix(TO)
SB, sb = compute_scale_matrix(TB)
ZOliveira = dot(so.T, so)
ZBrahimi = dot(sb.T, sb)
	
# MAIN
path_images = '../../Images_for_tests/Planar'
files = os.listdir(path_images)
n_images = 0
results = []

for file in tqdm(files):
	full_path = os.path.join(path_images, file)
	if os.path.isfile(full_path):
		image = around(255*imread(full_path, as_gray=True))
		h, w = image.shape
		A = Tools.umount(image, (8, 8))# - 128
		DctPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T, A), T.T)
		Aprime1Oliveira = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO, A), TO.T)
		Aprime1Brahimi = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB, A), TB.T)
		
		BUFFER = {'DCT': {'PSNR':[], 'SSIM':[], 'BPP':[]},
			'OLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]}, 
			'BRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]}}
		# Laço de processamento dos diferentes métodos
		for index, QF in enumerate(quality_factors):
			# Quantização padrão do JPEG
			QOliveira = quantize(QF, Q0)
			QBrahimi = quantize(QF, QB)
			
			QOliveiraRounded = np2_round(QOliveira)
			QBrahimiCeiled = np2_ceil(QBrahimi)

			QfOliveiraRounded = divide(QOliveiraRounded, ZOliveira)
			QbOliveiraRounded = multiply(QOliveiraRounded, ZOliveira)
			QfBrahimiCeiled = divide(QBrahimiCeiled, ZBrahimi)
			QbBrahimiCeiled = multiply(QBrahimiCeiled, ZBrahimi)


			## DCT
			DctPrime2 = multiply(around(divide(DctPrime1, QOliveira)), QOliveira)
			DctPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, DctPrime2), T)
			B = clip(Tools.remount(DctPrime3, (h, w)), 0, 255)
			DctPrime2 = DctPrime2.reshape(h, w)
			#DATAS['DCT']['PSNR'][index] += peak_signal_noise_ratio(image, B, data_range=255)
			#DATAS['DCT']['SSIM'][index] += structural_similarity(image, B, data_range=255)
			#DATAS['DCT']['BPP'][index] += bpp(DctPrime2)
			BUFFER['DCT']['PSNR'].append(peak_signal_noise_ratio(image, B, data_range=255))
			BUFFER['DCT']['SSIM'].append(structural_similarity(image, B, data_range=255))
			BUFFER['DCT']['BPP'].append(bpp(DctPrime2))

			# Proposta do Oliveira (TO|QO|NP2ROUND)
			QO_forward = tile(asarray([QfOliveiraRounded]), (Aprime1Oliveira.shape[0], 1, 1))
			QO_backward = tile(asarray([QbOliveiraRounded]), (Aprime1Oliveira.shape[0], 1, 1))
			OliveiraPrime2 = multiply(around(divide(Aprime1Oliveira, QO_forward)), QO_backward)
			OliveiraPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, OliveiraPrime2), TO)
			B = clip(Tools.remount(OliveiraPrime3, (h, w)), 0, 255) #+ 128
			OliveiraPrime2 = OliveiraPrime2.reshape(h, w)
			#DATAS['Oliveira-propose']['PSNR'][index] += peak_signal_noise_ratio(image, B, data_range=255)
			#DATAS['Oliveira-propose']['SSIM'][index] += structural_similarity(image, B, data_range=255)
			#DATAS['Oliveira-propose']['BPP'][index] += bpp(OliveiraPrime2)
			BUFFER['OLIVEIRA']['PSNR'].append(peak_signal_noise_ratio(image, B, data_range=255))
			BUFFER['OLIVEIRA']['SSIM'].append(structural_similarity(image, B, data_range=255))
			BUFFER['OLIVEIRA']['BPP'].append(bpp(OliveiraPrime2))
			
			# Proposta da Brahimi (TB|QB|NP2CEIL)
			QB_forward = tile(asarray([QfBrahimiCeiled]), (Aprime1Brahimi.shape[0], 1, 1))
			QB_backward = tile(asarray([QbBrahimiCeiled]), (Aprime1Brahimi.shape[0], 1, 1))
			BrahimiPrime2 = multiply(around(divide(Aprime1Brahimi, QB_forward)), QB_backward)
			BrahimiPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, BrahimiPrime2), TB)
			B = clip(Tools.remount(BrahimiPrime3, (h, w)), 0, 255) #+ 128
			BrahimiPrime2 = BrahimiPrime2.reshape(h, w)
			#DATAS['Brahimi-propose']['PSNR'][index] += peak_signal_noise_ratio(image, B, data_range=255)
			#DATAS['Brahimi-propose']['SSIM'][index] += structural_similarity(image, B, data_range=255)
			#DATAS['Brahimi-propose']['BPP'][index] += bpp(BrahimiPrime2)
			BUFFER['BRAHIMI']['PSNR'].append(peak_signal_noise_ratio(image, B, data_range=255))
			BUFFER['BRAHIMI']['SSIM'].append(structural_similarity(image, B, data_range=255))
			BUFFER['BRAHIMI']['BPP'].append(bpp(BrahimiPrime2))

		results.append({'File name':file, 'Method':'DCT', 'PSNR':BUFFER['DCT']['PSNR'], 'SSIM':BUFFER['DCT']['SSIM'], 'BPP':BUFFER['DCT']['BPP']})
		results.append({'File name':file, 'Method':'Oliveira', 'PSNR':BUFFER['OLIVEIRA']['PSNR'], 'SSIM':BUFFER['OLIVEIRA']['SSIM'], 'BPP':BUFFER['OLIVEIRA']['BPP']})
		results.append({'File name':file, 'Method':'Brahimi', 'PSNR':BUFFER['BRAHIMI']['PSNR'], 'SSIM':BUFFER['BRAHIMI']['SSIM'], 'BPP':BUFFER['BRAHIMI']['BPP']})

		n_images += 1

results = sorted(results, key=itemgetter('File name'))
fieldnames = ['File name', 'Method', 'PSNR', 'SSIM', 'BPP']
with open('standards.csv', 'w') as csv_file:
	writer = csv.DictWriter(csv_file, fieldnames)
	writer.writeheader()
	for result in results:
		writer.writerow(result)

'''
# Laço para o cálculo das médias
for key in DATAS:
	data = DATAS[key]
	for metric in data:
		data[metric] = asarray(data[metric]) / n_images

# Plotagem dos gráficos de curvas
plot_cols, plot_rows = (2, 2)
fig, axes = plot.subplots(plot_rows, plot_cols, label="Oliveira propose")
for row_index_plot in range(plot_rows):
	for col_index_plot in range(plot_cols):
		axes[row_index_plot, col_index_plot].grid(True)

		if row_index_plot == 0 and col_index_plot == 0:
			axes[row_index_plot, col_index_plot].set_title("QF vs. PSNR"); axes[row_index_plot, col_index_plot].set_xlabel("QF values"); axes[row_index_plot, col_index_plot].set_ylabel("PSNR values")
			for data_key in DATAS.keys():
				data = DATAS[data_key]
				color, style, legend = ('', '', '')
				for method_key in METHODS.values():
					if data_key == method_key['Label']:
						color = method_key['Color']; style = method_key['Style']; legend = method_key['Legend']
				axes[row_index_plot, col_index_plot].plot(quality_factors, data['PSNR'], color=color, ls=style, label=legend)

		if row_index_plot == 0 and col_index_plot == 1:
			axes[row_index_plot, col_index_plot].set_xlabel("QF values"); axes[row_index_plot, col_index_plot].set_ylabel("SSIM values")
			for data_key in DATAS.keys():
				data = DATAS[data_key]
				color, style, legend = ('', '', '')
				for method_key in METHODS.values():
					if data_key == method_key['Label']:
						color = method_key['Color']; style = method_key['Style']; legend = method_key['Legend']
				axes[row_index_plot, col_index_plot].plot(quality_factors, data['SSIM'], color=color, ls=style, label=legend)

		if row_index_plot == 1 and col_index_plot == 0:
			axes[row_index_plot, col_index_plot].set_title("BPP vs. PSNR"); axes[row_index_plot, col_index_plot].set_xlabel("BPP values"); axes[row_index_plot, col_index_plot].set_ylabel("PSNR values")
			for data_key in DATAS.keys():
				data = DATAS[data_key]
				color, style, legend = ('', '', '')
				for method_key in METHODS.values():
					if data_key == method_key['Label']:
						color = method_key['Color']; style = method_key['Style']; legend = method_key['Legend']
				axes[row_index_plot, col_index_plot].plot(data['BPP'], data['PSNR'], color=color, ls=style, label=legend)

		if row_index_plot == 1 and col_index_plot == 1:
			axes[row_index_plot, col_index_plot].set_title("BPP vs. SSIM"); axes[row_index_plot, col_index_plot].set_xlabel("BPP values"); axes[row_index_plot, col_index_plot].set_ylabel("SSIM values")
			for data_key in DATAS.keys():
				data = DATAS[data_key]
				color, style, legend = ('', '', '')
				for method_key in METHODS.values():
					if data_key == method_key['Label']:
						color = method_key['Color']; style = method_key['Style']; legend = method_key['Legend']
				axes[row_index_plot, col_index_plot].plot(data['BPP'], data['SSIM'], color=color, ls=style, label=legend)

		axes[row_index_plot, col_index_plot].legend()
fig.tight_layout()
plot.show()
'''