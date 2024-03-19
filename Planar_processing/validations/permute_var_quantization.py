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
    'RDCT-Oliveira': {'Label': 'RDCT-Oliveira', 'Color': 'black', 'Style': 'dashed', 'Legend': 'RDCT-Oliveira'},
    'RDCT-Brahimi': {'Label': 'RDCT-Brahimi', 'Color': 'black', 'Style': 'dotted', 'Legend': 'RDCT-Brahimi'},
    'Oliveira-propose': {'Label': 'Oliveira-propose', 'Color': 'r', 'Style': 'dashed', 'Legend': 'Oliveira-propose'},
    'PERMUTACAO1': {'Label': 'PERMUTACAO1', 'Color': 'g', 'Style': 'solid', 'Legend': 'QB'},
	'PERMUTACAO2': {'Label': 'PERMUTACAO2', 'Color': 'yellow', 'Style': 'solid', 'Legend': 'NP2_ceil'},
	'PERMUTACAO3': {'Label': 'PERMUTACAO3', 'Color': 'b', 'Style': 'solid', 'Legend': 'TB'}

}
DATAS = {method: {'PSNR': zeros(len(quality_factors)), 'SSIM': zeros(len(quality_factors)), 'BPP': zeros(len(quality_factors))} for method in METHODS}
T = calculate_matrix_of_transformation(8)
SO, so = compute_scale_matrix(TO)
SB, sb = compute_scale_matrix(TB)
ZOliveira = dot(so.T, so)
ZBrahimi = dot(sb.T, sb)
	
# MAIN
path_images = '../Images_for_tests/Planar'
files = os.listdir(path_images)
n_images = 0
for file in tqdm(files[:3]):
	full_path = os.path.join(path_images, file)
	if os.path.isfile(full_path):
		image = around(255*imread(full_path, as_gray=True))
		h, w = image.shape
		A = Tools.umount(image, (8, 8))# - 128
		Aprime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO, A), TO.T)

		# Laço de processamento dos diferentes métodos
		for index, QF in enumerate(quality_factors):
			# Quantização padrão do JPEG
			QOliveira = quantize(QF, Q0)
			QBrahimi = quantize(QF, QB)
			# Quantização padrão do JPEG com aproximação
			QOliveira_f = divide(QOliveira, ZOliveira)
			QOliveira_b = multiply(QOliveira, ZOliveira)
			# Quantização Brahimi com aproximação
			QBrahimi_f = divide(QBrahimi, ZBrahimi)
			QBrahimi_b = multiply(QBrahimi, ZBrahimi)

			QOliveiraRounded = np2_round(QOliveira)
			QOliveiraCeiled = np2_ceil(QOliveira)
			QBrahimiRounded = np2_round(QBrahimi)
			QBrahimiCeiled = np2_ceil(QBrahimi)

			QfOliveiraRounded = divide(QOliveiraRounded, ZOliveira)
			QbOliveiraRounded = multiply(QOliveiraRounded, ZOliveira)
			QfOliveiraCeiled = divide(QOliveiraCeiled, ZOliveira)
			QbOliveiraCeiled = multiply(QOliveiraCeiled, ZOliveira)

			QfBrahimiRounded = divide(QBrahimiRounded, ZBrahimi)
			QbBrahimiRounded = multiply(QBrahimiRounded, ZBrahimi)
			QfBrahimiCeiled = divide(QBrahimiCeiled, ZBrahimi)
			QbBrahimiCeiled = multiply(QBrahimiCeiled, ZBrahimi)


			## DCT
			DctPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T, A), T.T)
			DctPrime2 = multiply(around(divide(DctPrime1, QOliveira)), QOliveira)
			DctPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, DctPrime2), T)
			B = clip(Tools.remount(DctPrime3, (h, w)), 0, 255)
			DctPrime2 = DctPrime2.reshape(h, w)
			DATAS['DCT']['PSNR'][index] += peak_signal_noise_ratio(image, B, data_range=255)
			DATAS['DCT']['SSIM'][index] += structural_similarity(image, B, data_range=255)
			DATAS['DCT']['BPP'][index] += count_nonzero(logical_not(isclose(DctPrime2, 0))) * 8 / (DctPrime2.shape[0]*DctPrime2.shape[1])

			# RDCT Oliveira
			Q_forward = tile(asarray([QOliveira_f]), (Aprime1.shape[0], 1 , 1))
			Q_backward = tile(asarray([QOliveira_b]), (Aprime1.shape[0], 1 , 1))
			RdctOliveiraPrime2 = multiply(around(divide(Aprime1, Q_forward)), Q_backward)
			RdctOliveiraPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, RdctOliveiraPrime2), TO)
			B = clip(Tools.remount(RdctOliveiraPrime3, (h, w)), 0, 255) #+ 128
			RdctOliveiraPrime2 = RdctOliveiraPrime2.reshape(h, w)
			DATAS['RDCT-Oliveira']['PSNR'][index] += peak_signal_noise_ratio(image, B, data_range=255)
			DATAS['RDCT-Oliveira']['SSIM'][index] += structural_similarity(image, B, data_range=255)
			DATAS['RDCT-Oliveira']['BPP'][index] += count_nonzero(logical_not(isclose(RdctOliveiraPrime2, 0))) * 8 / (RdctOliveiraPrime2.shape[0]*RdctOliveiraPrime2.shape[1])

			# RDCT Brahimi
			Q_forward = tile(asarray([QBrahimi_f]), (Aprime1.shape[0], 1 , 1))
			Q_backward = tile(asarray([QBrahimi_b]), (Aprime1.shape[0], 1 , 1))
			RdctBrahimiPrime2 = multiply(around(divide(Aprime1, Q_forward)), Q_backward)
			RdctBrahimiPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, RdctBrahimiPrime2), TB)
			B = clip(Tools.remount(RdctBrahimiPrime3, (h, w)), 0, 255) #+ 128
			RdctBrahimiPrime2 = RdctBrahimiPrime2.reshape(h, w)
			DATAS['RDCT-Brahimi']['PSNR'][index] += peak_signal_noise_ratio(image, B, data_range=255)
			DATAS['RDCT-Brahimi']['SSIM'][index] += structural_similarity(image, B, data_range=255)
			DATAS['RDCT-Brahimi']['BPP'][index] += count_nonzero(logical_not(isclose(RdctBrahimiPrime2, 0))) * 8 / (RdctBrahimiPrime2.shape[0]*RdctBrahimiPrime2.shape[1])

			# Proposta do oliveira (TO|QO|NP2ROUND)
			Q_forward = tile(asarray([QfOliveiraRounded]), (Aprime1.shape[0], 1, 1))
			Q_backward = tile(asarray([QbOliveiraRounded]), (Aprime1.shape[0], 1, 1))
			OliveiraPrime2 = multiply(around(divide(Aprime1, Q_forward)), Q_backward)
			OliveiraPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, OliveiraPrime2), TO)
			B = clip(Tools.remount(OliveiraPrime3, (h, w)), 0, 255) #+ 128
			OliveiraPrime2 = OliveiraPrime2.reshape(h, w)
			DATAS['Oliveira-propose']['PSNR'][index] += peak_signal_noise_ratio(image, B, data_range=255)
			DATAS['Oliveira-propose']['SSIM'][index] += structural_similarity(image, B, data_range=255)
			DATAS['Oliveira-propose']['BPP'][index] += count_nonzero(logical_not(isclose(OliveiraPrime2, 0))) * 8 / (OliveiraPrime2.shape[0]*OliveiraPrime2.shape[1])

			# Permutação da matriz de quantização: np2_round(QB)
			Q_forward = tile(asarray([QfBrahimiRounded]), (Aprime1.shape[0], 1, 1))
			Q_backward = tile(asarray([QbBrahimiRounded]), (Aprime1.shape[0], 1, 1))
			Permutacao1Prime2 = multiply(around(divide(Aprime1, Q_forward)), Q_backward)
			Permutacao1Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, Permutacao1Prime2), TO)
			B = clip(Tools.remount(Permutacao1Prime3, (h, w)), 0, 255) #+ 128
			Permutacao1Prime2 = Permutacao1Prime2.reshape(h, w)
			DATAS['PERMUTACAO1']['PSNR'][index] += peak_signal_noise_ratio(image, B, data_range=255)
			DATAS['PERMUTACAO1']['SSIM'][index] += structural_similarity(image, B, data_range=255)
			DATAS['PERMUTACAO1']['BPP'][index] += count_nonzero(logical_not(isclose(Permutacao1Prime2, 0))) * 8 / (Permutacao1Prime2.shape[0]*Permutacao1Prime2.shape[1])

			# Permutação do np2: np2_ceil(QO)
			Q_forward = tile(asarray([QfOliveiraCeiled]), (Aprime1.shape[0], 1, 1))
			Q_backward = tile(asarray([QbOliveiraCeiled]), (Aprime1.shape[0], 1, 1))
			Permutacao2Prime2 = multiply(around(divide(Aprime1, Q_forward)), Q_backward)
			Permutacao2Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, Permutacao2Prime2), TO)
			B = clip(Tools.remount(Permutacao2Prime3, (h, w)), 0, 255) #+ 128
			Permutacao2Prime2 = Permutacao2Prime2.reshape(h, w)
			DATAS['PERMUTACAO2']['PSNR'][index] += peak_signal_noise_ratio(image, B, data_range=255)
			DATAS['PERMUTACAO2']['SSIM'][index] += structural_similarity(image, B, data_range=255)
			DATAS['PERMUTACAO2']['BPP'][index] += count_nonzero(logical_not(isclose(Permutacao2Prime2, 0))) * 8 / (Permutacao2Prime2.shape[0]*Permutacao2Prime2.shape[1])

			# Permutação da matriz de transformacao: np2_round(QB)
			Q_forward = tile(asarray([QfOliveiraRounded]), (Aprime1.shape[0], 1, 1))
			Q_backward = tile(asarray([QbOliveiraRounded]), (Aprime1.shape[0], 1, 1))
			Permutacao3Prime2 = multiply(around(divide(Aprime1, Q_forward)), Q_backward)
			Permutacao3Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, Permutacao3Prime2), TB)
			B = clip(Tools.remount(Permutacao3Prime3, (h, w)), 0, 255) #+ 128
			Permutacao3Prime2 = Permutacao3Prime2.reshape(h, w)
			DATAS['PERMUTACAO3']['PSNR'][index] += peak_signal_noise_ratio(image, B, data_range=255)
			DATAS['PERMUTACAO3']['SSIM'][index] += structural_similarity(image, B, data_range=255)
			DATAS['PERMUTACAO3']['BPP'][index] += count_nonzero(logical_not(isclose(Permutacao3Prime2, 0))) * 8 / (Permutacao3Prime2.shape[0]*Permutacao3Prime2.shape[1])

			#
		n_images += 1

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
