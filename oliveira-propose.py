import os
from matrix import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
from scipy import signal
from pdb import set_trace as pause
from matplotlib import pyplot as plot
from time import time
from tools import *
from datetime import datetime
from tqdm import tqdm

""" Definição das funções auxiliares """
def quantize(quality_factor:int, quantization_matrix:ndarray) -> ndarray:
	s = 0.0
	if quality_factor < 50:
		s = 5_000 / quality_factor
	else:
		s = 200 - (2 * quality_factor)
	resulting_matrix = floor((s * quantization_matrix + 50) / 100)
	return resulting_matrix

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

def map_k_and_el(row_index:int, image_height:int) -> tuple:
	el = row_index/image_height * pi - pi/2
	kprime = arange(8)
	k = clip(0, 7, around(kprime/cos(el))).astype(int)
	return (k, el)

def build_LUT(image_height:int, N:int=8):
	ks, els = [], []
	for row_index in range(0, image_height+1, N):
		k, el = map_k_and_el(row_index, image_height)
		ks.append(k); els.append(el)
	ks = asarray(ks); els = abs(asarray(els))
	k_LUT = unique(ks, axis=0)
	min_LUT = asarray([finfo('f').max for x in k_LUT])
	max_LUT = []
	aux_max_LUT = [finfo('f').min for x in k_LUT]
	for idx in range(len(ks)):
		for idx2 in range(len(k_LUT)):
			if sum(ks[idx] - k_LUT[idx2]) == 0:
				if els[idx] > aux_max_LUT[idx2]: 
					aux_max_LUT[idx2] = els[idx]
				if els[idx] < min_LUT[idx2]: 
					min_LUT[idx2] = els[idx] 
	for idx in range(len(k_LUT)):
		if idx != len(k_LUT)-1: 
			max_LUT.append(min_LUT[idx+1])
		else: 
			max_LUT.append(aux_max_LUT[idx])
	max_LUT = asarray(max_LUT)
	return k_LUT, min_LUT, max_LUT

def QtildeAtEl(k_lut:ndarray, min_lut:ndarray, max_lut:ndarray, el:float32, quantization_matrix:ndarray, QF:int= 50):
		ks = None
		el = abs(el) # LUT is mirrored
		for idx in range(len(k_lut)):
			if el >= min_lut[idx] and el < max_lut[idx]: 
				ks = k_lut[idx]
		if ks is None and el == 0: ks = k_lut[0]
		Q = vstack(([quantization_matrix[:,k] for k in ks])).T
		Q = reshape(Q, quantization_matrix.shape)
		return asarray(Q) # É transposto?

def prepareQPhi(image:ndarray, quantization_matrix:ndarray, QF:int=50, N = 8):
	h, w = image.shape
	k_lut, min_lut, max_lut = build_LUT(h)
	els = linspace(-pi/2, pi/2, h//N+1)
	els = 0.5*(els[1:] + els[:-1]) # gets the "central" block elevation
	QPhi = []
	for el in els:
		QPhi.append(QtildeAtEl(k_lut, min_lut, max_lut, el, quantization_matrix, QF))
	QPhi = asarray(QPhi)
	QPhi = repeat(QPhi, w//N, axis=0)
	#plot.imshow(tools.Tools.remount(QPhi, (h, w))); plot.show() # plot the quantization matrices map
	return QPhi

def np2_round(quantization_matrix:matrix) -> matrix:
	return power(2, log2(quantization_matrix).round())
"""Função de transformação de uma matriz em uma matriz de potências de dois - Oliveira """

def np2_ceil(quantization_matrix:matrix) -> matrix:
	return power(2, ceil(log2(quantization_matrix)))
"""Função de transformação de uma matriz em uma matriz de potências de dois - Brahimi """

def compute_scale_matrix(transformation_matrix:ndarray) -> matrix:
	if transformation_matrix.shape != (8,8):
		print("Erro: matrix de trasformação deve ser 8x8 ")
	else:
		values = []
		for row in range(8):
			count = 0
			for col in range(8):
				if transformation_matrix[row,col] != 0:
					count += 1
			values.append(1/sqrt(count))
		scale_matrix = matrix(diag(values)).T
		return scale_matrix, matrix(diag(scale_matrix))	# Matrix diagonal e elementos da matriz diagonal vetorizados

def WSSSIM(img1, img2, K1 = .01, K2 = .03, L = 255):

    def __fspecial_gauss(size, sigma):
        x, y = mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    def __weights(height, width):
        deltaTheta = 2*pi/width 
        column = asarray([cos( deltaTheta * (j - height/2.+0.5)) for j in range(height)])
        return repeat(column[:, newaxis], width, 1)

    img1 = float64(img1)
    img2 = float64(img2)

    k = 11
    sigma = 1.5
    window = __fspecial_gauss(k, sigma)
    window2 = zeros_like(window); window2[k//2,k//2] = 1 
 
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    sigma1_sq = signal.convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1*img2, window, 'valid') - mu1_mu2
   
    W = __weights(*img1.shape)
    Wi = signal.convolve2d(W, window2, 'valid')

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)) * Wi
    mssim = sum(ssim_map)/sum(Wi)

    return mssim


def WSPSNR(img1, img2, max = 255.): # img1 e img2 devem ter shape hx2h e ser em grayscale; max eh o maximo valor possivel em img1 e img2 (verificar se deve ser 1 ou 255)

   def __weights(height, width):
      phis = arange(height+1)*pi/height
      deltaTheta = 2*pi/width 
      column = asarray([deltaTheta * (-cos(phis[j+1])+cos(phis[j])) for j in range(height)])
      return repeat(column[:, newaxis], width, 1)

   w = __weights(*img1.shape)
   # from matplotlib import pyplot as plt; plt.imshow(w); plt.show()
   wmse = sum((img1-img2)**2*w)/(4*pi) # É ASSIM MESMO
   return 10*log10(max**2/wmse)

""" Estruturas de armazenamento para ratios """
quality_factors = range(5, 100, 5)
DCT = {'Label':'DCT' ,'PSNR':[], 'SSIM':[], 'BPP':[], 'Color':'black', 'Style':'solid', 'Legend':'DCT'}																# Aplicação da DCT
RDCT = {'Label':'RDCT' ,'PSNR':[], 'SSIM':[], 'BPP':[], 'Color':'black', 'Style':'dashed', 'Legend':'RDCT'}															# Aproximação da DCT
OLIVEIRA1 = {'Label':'OLIVEIRA1' ,'PSNR':[], 'SSIM':[], 'BPP':[], 'Color':'g', 'Style':'solid', 'Legend':'$\operatorname{np2}((\mathbf{Q})_\phi\oslash\mathbf{Z})$'} 		# np2(phi(Q)/Z) Ideal do ponto de vista teórico/matemático
OLIVEIRA2 = {'Label':'OLIVEIRA2' ,'PSNR':[], 'SSIM':[], 'BPP':[], 'Color':'r', 'Style':'solid', 'Legend':'$\operatorname{np2}((\mathbf{Q})\oslash\mathbf{Z})_\phi$'} 		# np2(phi(Q/Z)) Ideal do ponto de vista de implementação
OLIVEIRA3 = {'Label':'OLIVEIRA3' ,'PSNR':[], 'SSIM':[], 'BPP':[], 'Color':'b', 'Style':'dashed', 'Legend':'$(\operatorname{np2}(\mathbf{Q})\oslash\mathbf{Z})_\phi$'} 	# phi(np2(Q/Z)) Ideal do ponto de vista de implementação
METHODS = [DCT, RDCT, OLIVEIRA1, OLIVEIRA2, OLIVEIRA3]
DATAS = {
    'DCT': {'PSNR': zeros(len(quality_factors)), 'SSIM': zeros(len(quality_factors)), 'BPP': zeros(len(quality_factors))},
    'RDCT': {'PSNR': zeros(len(quality_factors)), 'SSIM': zeros(len(quality_factors)), 'BPP': zeros(len(quality_factors))},
    'OLIVEIRA1': {'PSNR': zeros(len(quality_factors)), 'SSIM': zeros(len(quality_factors)), 'BPP': zeros(len(quality_factors))},
    'OLIVEIRA2': {'PSNR': zeros(len(quality_factors)), 'SSIM': zeros(len(quality_factors)), 'BPP': zeros(len(quality_factors))},
    'OLIVEIRA3': {'PSNR': zeros(len(quality_factors)), 'SSIM': zeros(len(quality_factors)), 'BPP': zeros(len(quality_factors))}
}


""" Pré processamento de matrizes e vetores """
T = calculate_matrix_of_transformation(8)
s_matrix = matrix(sqrt(linalg.inv(dot(TO, TO.T))))
s_vector = matrix(diag(s_matrix))
Z1 = dot(s_vector.T, s_vector)

# Início do temporizador
now = datetime.now()
hour = now.hour
minute = now.minute
second = now.second
t_start = time()

""" Carregamento da(s) imagem(ns) de teste """
path_images = "test_images/4K"
print(f"O processamento começou as {hour} horas, {minute} minutos e {second} segundos")
file_names = os.listdir(path_images)
n_images = 0
for file_name in tqdm(file_names, mininterval=0.00001):
	full_path = os.path.join(path_images, file_name)
	if os.path.isfile(full_path):
		image = around(255*imread(full_path, as_gray=True))
		h, w = image.shape
		A = Tools.umount(image, (8, 8))# - 128
		Aprime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO, A), TO.T) # forward transform

		""" Laço de Processamento """
		for index, QF in enumerate(quality_factors):

			Q = quantize(QF, Q0)
			Q_f1 = divide(Q, Z1)
			Q_b1 = multiply(Z1, Q)

			# DCT TODO: Aplicar Phi(Q)
			QPhi = quantize(QF, prepareQPhi(image, Q0, QF))
			DctPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T, A), T.T) 					# forward transform
			DctPrime2 = multiply(around(divide(DctPrime1, QPhi)), QPhi)									# quantization encoding & decoding
			DctPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, DctPrime2), T) 			# inverse transform
			B = clip(Tools.remount(DctPrime3, (h, w)), 0, 255)
			DctPrime2 = DctPrime2.reshape(h, w)
			DATAS['DCT']['PSNR'][index] += WSPSNR(image, B)
			DATAS['DCT']['SSIM'][index] += WSSSIM(image, B)
			DATAS['DCT']['BPP'][index] += count_nonzero(logical_not(isclose(DctPrime2, 0))) * 8 / (DctPrime2.shape[0]*DctPrime2.shape[1])
			#plot.imshow(B, cmap='gray'); plot.title(f"DCT - Qf = {QF}") ;plot.show()

			# RDCT TODO: Aplicar Phi(Q)
			Z = tile(asarray([Z1]), (Aprime1.shape[0], 1, 1))
			QPhi_forward = divide(QPhi, Z)
			QPhi_backward = multiply(Z, QPhi)
			Aprime2 = multiply(around(divide(Aprime1, QPhi_forward)), QPhi_backward) 					# quantization
			Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, Aprime2), TO) 			# inverse transform
			B = clip(Tools.remount(Aprime3, (h, w)), 0, 255) #+ 128
			Aprime2 = Aprime2.reshape(h, w)
			DATAS['RDCT']['PSNR'][index] += WSPSNR(image, B)
			DATAS['RDCT']['SSIM'][index] += WSSSIM(image, B)
			DATAS['RDCT']['BPP'][index] += count_nonzero(logical_not(isclose(Aprime2, 0))) * 8 / (Aprime2.shape[0]*Aprime2.shape[1])
			#plot.imshow(B, cmap='gray'); plot.title(f"RDCT - Qf = {QF}") ;plot.show()

			# QPhi = np2(phi(Q)/Z)
			QPhi_forward = divide(QPhi, Z)
			QPhi_backward = multiply(Z, QPhi)
			Aprime2 = multiply(around(divide(Aprime1, QPhi_forward)), QPhi_backward)					# quantization
			Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, Aprime2), TO) 			# inverse transform
			B = clip(Tools.remount(Aprime3, (h, w)), 0, 255) #+ 128
			Aprime2 = Aprime2.reshape(h, w)
			DATAS['OLIVEIRA1']['PSNR'][index] += WSPSNR(image, B)
			DATAS['OLIVEIRA1']['SSIM'][index] += WSSSIM(image, B)
			DATAS['OLIVEIRA1']['BPP'][index] += count_nonzero(logical_not(isclose(Aprime2, 0))) * 8 / (Aprime2.shape[0]*Aprime2.shape[1])
			#plot.imshow(B, cmap='gray'); plot.title(f"np2(phi(Q)/Z) - Qf = {QF}") ;plot.show()

			# QPhi = phi(np2(Q/Z))
			Q_forward = np2_round(divide(Q, Z1))
			Q_backward = np2_round(multiply(Z1, Q))
			QPhi_forward = prepareQPhi(image, Q_forward, QF)
			QPhi_backward = prepareQPhi(image, Q_backward, QF)
			Aprime2 = multiply(around(divide(Aprime1, QPhi_forward)), QPhi_backward) 					# quantization
			Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, Aprime2), TO) 			# inverse transform
			B = clip(Tools.remount(Aprime3, (h, w)), 0, 255) #+ 128
			Aprime2 = Aprime2.reshape(h, w)
			DATAS['OLIVEIRA2']['PSNR'][index] += WSPSNR(image, B)
			DATAS['OLIVEIRA2']['SSIM'][index] += WSSSIM(image, B)
			DATAS['OLIVEIRA2']['BPP'][index] += count_nonzero(logical_not(isclose(Aprime2, 0))) * 8 / (Aprime2.shape[0]*Aprime2.shape[1])
			#plot.imshow(B, cmap='gray'); plot.title(f"phi(np2(Q/Z)) - Qf = {QF}") ;plot.show()

			# QPhi = np2(phi(Q/Z))
			Q_forward = divide(Q, Z1)
			Q_backward = multiply(Z1, Q)
			QPhi_forward = np2_round(prepareQPhi(image, Q_forward, QF))
			QPhi_backward = np2_round(prepareQPhi(image, Q_backward, QF))
			Aprime2 = multiply(around(divide(Aprime1, QPhi_forward)), QPhi_backward) 					# quantization
			Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, Aprime2), TO) 			# inverse transform
			B = clip(Tools.remount(Aprime3, (h, w)), 0, 255) #+ 128
			Aprime2 = Aprime2.reshape(h, w)
			DATAS['OLIVEIRA3']['PSNR'][index] += WSPSNR(image, B)
			DATAS['OLIVEIRA3']['SSIM'][index] += WSSSIM(image, B)
			DATAS['OLIVEIRA3']['BPP'][index] += count_nonzero(logical_not(isclose(Aprime2, 0))) * 8 / (Aprime2.shape[0]*Aprime2.shape[1])
			#plot.imshow(B, cmap='gray'); plot.title(f"np2(phi(Q/Z)) - Qf = {QF}") ;plot.show()
		n_images += 1

# Cálculo do tempo de processamento
t_end = time()
now = datetime.now()
processment_time = t_end - t_start
horas = processment_time // 3600
minutos = (processment_time % 3600) // 60
segundos = (processment_time % 3600) % 60
print(f"Tempo de processamento: {int(horas)} horas, {int(minutos)} minutos e {int(segundos)} segundos.")
print(f"O processamento terminou as {now.hour} horas, {now.minute} minutos e {now.second} segundos")

# Laço para o cálculo das médias
for key in DATAS:
	data = DATAS[key]
	for metric in data:
		data[metric] = asarray(data[metric]) / n_images

# Plotagem dos gráficos de curvas
plot_cols, plot_rows = (2, 2)
fig, axes = plot.subplots(plot_rows, plot_cols)
for row_index_plot in range(plot_rows):
	for col_index_plot in range(plot_cols):
		axes[row_index_plot, col_index_plot].grid(True)

		if row_index_plot == 0 and col_index_plot == 0:
			axes[row_index_plot, col_index_plot].set_title("QF vs. PSNR"); axes[row_index_plot, col_index_plot].set_xlabel("QF values"); axes[row_index_plot, col_index_plot].set_ylabel("PSNR values")
			for data_key in DATAS.keys():
				data = DATAS[data_key]
				color, style, legend = ('', '', '')
				for method_key in METHODS:
					if data_key == method_key['Label']:
						color = method_key['Color']; style = method_key['Style']; legend = method_key['Legend']
				axes[row_index_plot, col_index_plot].plot(quality_factors, data['PSNR'], color=color, ls=style, label=legend)

		if row_index_plot == 0 and col_index_plot == 1:
			axes[row_index_plot, col_index_plot].set_xlabel("QF values"); axes[row_index_plot, col_index_plot].set_ylabel("SSIM values")
			for data_key in DATAS.keys():
				data = DATAS[data_key]
				color, style, legend = ('', '', '')
				for method_key in METHODS:
					if data_key == method_key['Label']:
						color = method_key['Color']; style = method_key['Style']; legend = method_key['Legend']
				axes[row_index_plot, col_index_plot].plot(quality_factors, data['SSIM'], color=color, ls=style, label=legend)

		if row_index_plot == 1 and col_index_plot == 0:
			axes[row_index_plot, col_index_plot].set_title("BPP vs. PSNR"); axes[row_index_plot, col_index_plot].set_xlabel("BPP values"); axes[row_index_plot, col_index_plot].set_ylabel("PSNR values")
			for data_key in DATAS.keys():
				data = DATAS[data_key]
				color, style, legend = ('', '', '')
				for method_key in METHODS:
					if data_key == method_key['Label']:
						color = method_key['Color']; style = method_key['Style']; legend = method_key['Legend']
				axes[row_index_plot, col_index_plot].plot(data['BPP'], data['PSNR'], color=color, ls=style, label=legend)

		if row_index_plot == 1 and col_index_plot == 1:
			axes[row_index_plot, col_index_plot].set_title("BPP vs. SSIM"); axes[row_index_plot, col_index_plot].set_xlabel("BPP values"); axes[row_index_plot, col_index_plot].set_ylabel("SSIM values")
			for data_key in DATAS.keys():
				data = DATAS[data_key]
				color, style, legend = ('', '', '')
				for method_key in METHODS:
					if data_key == method_key['Label']:
						color = method_key['Color']; style = method_key['Style']; legend = method_key['Legend']
				axes[row_index_plot, col_index_plot].plot(data['BPP'], data['SSIM'], color=color, ls=style, label=legend)

		axes[row_index_plot, col_index_plot].legend()
fig.tight_layout()
plot.show()


