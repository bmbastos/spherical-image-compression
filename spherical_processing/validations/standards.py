import os
import csv
import pandas as pd
from numpy import *
from time import time
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import signal
from matplotlib import pyplot as plot
from pdb import set_trace as pause
from tqdm import tqdm
from operator import itemgetter
from matrixes import *
from tools import *

# DEFINIÇÃO DE FUNÇÕES

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

# Aplicação do fator de quantização com base no QF (Quality-factor) e na matriz usada
def quantize(quality_factor:int, quantization_matrix:ndarray) -> ndarray:
	s = 0.0
	if quality_factor < 50:
		s = 5_000 / quality_factor
	else:
		s = 200 - (2 * quality_factor)
	resulting_matrix = floor((s * quantization_matrix + 50) / 100)
	return resulting_matrix
""" Calcula a matriz de quantização dado um fator de quantização """

def bpp(quantized_image:ndarray):
	return count_nonzero(logical_not(isclose(quantized_image, 0))) * 8 / (quantized_image.shape[0]*quantized_image.shape[1])
""" Calcula a quantos bits são necessário por pixel"""

def compute_scale_matrix(transformation_matrix:ndarray) -> matrix:
	scale_matrix = matrix(sqrt(linalg.inv(dot(transformation_matrix, transformation_matrix.T))))
	scale_vector = matrix(diag(scale_matrix))
	return scale_matrix, scale_vector
""" Matrix diagonal e elementos da matriz diagonal vetorizados """

def np2_round(quantization_matrix:matrix) -> matrix:
	return power(2, around(log2(quantization_matrix)))
""" Função que calcula as potencias de dois mais próximas de uma dada matriz - Oliveira """

def np2_ceil(quantization_matrix:matrix) -> matrix:
	return power(2, ceil(log2(quantization_matrix))) # Não usar
"""Função de transformação de uma matriz em uma matriz de potências de dois - Brahimi """

def map_k_and_el(row_index:int, image_height:int) -> tuple:
	el = row_index/image_height * pi - pi/2
	kprime = arange(8)
	k = clip(0, 7, around(kprime/cos(el))).astype(int)
	return (k, el)

# Constroi a Look-up-table da imagem com base em sua altura
def build_LUT(image_height:int, N:int=8) -> tuple:
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

def printLUT(k_lut:ndarray, min_lut:ndarray, max_lut:ndarray):
        for idx in range(len(k_lut)):
            print(k_lut[idx], "%.4f" % min_lut[idx], "%.4f" % max_lut[idx])

def QtildeAtEl(k_lut:ndarray, min_lut:ndarray, max_lut:ndarray, el:float32, quantization_matrix:ndarray, QF:int= 50):
		ks = None
		Q = []
		QM = quantize(QF, quantization_matrix)
		el = abs(el) # LUT is mirrored
		for idx in range(len(k_lut)):
			if el >= min_lut[idx] and el < max_lut[idx]: 
				ks = k_lut[idx]
		if ks is None and isclose(el, 0): ks = k_lut[0]
		for k in ks:
			Q.append(QM.T.tolist()[k])
		Q = asarray(Q)
		Q = Q.T
		return Q

def encodeQuantiseNDecode(image:ndarray, transformation_matrix:ndarray, quantization_matrix:ndarray, N = 8):
	h, w = image.shape
	A = Tools.umount(image, (N, N))# - 128
	Aprime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', transformation_matrix, A), transformation_matrix.T) # forward transform
	Aprime2 = multiply(quantization_matrix, around(divide(Aprime1, quantization_matrix))) # quantization
	Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', transformation_matrix.T, Aprime2), transformation_matrix) # inverse transform
	B = Tools.remount(Aprime3, (h, w)) #+ 128
	return Aprime2.reshape(h,w), B 

def encodeQuantiseNDecodeBrahimi(image:ndarray, transformation_matrix:ndarray, quantization_matrix:ndarray, diagonal_matrix:ndarray, apply_np2:bool, np2:str='O', N = 8):
	h, w = image.shape
	diag_matrix = tile(asarray([diagonal_matrix]), (quantization_matrix.shape[0], 1, 1))
	quantization_matrix_forward = divide(quantization_matrix, diag_matrix)
	quantization_matrix_backward = multiply(diag_matrix, quantization_matrix)
	if apply_np2:
		if np2 == 'O':
			quantization_matrix_forward = np2_round(quantization_matrix_forward)
			quantization_matrix_backward = np2_round(quantization_matrix_backward)
		else:
			quantization_matrix_forward = np2_ceil(quantization_matrix_forward)
			quantization_matrix_backward = np2_ceil(quantization_matrix_backward)
	A = Tools.umount(image, (N, N))# - 128
	Aprime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', transformation_matrix, A), transformation_matrix.T) # forward transform
	Aprime2 = multiply(divide(Aprime1, quantization_matrix_forward).round(), quantization_matrix_backward) # quantization
	Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', transformation_matrix.T, Aprime2), transformation_matrix) # inverse transform
	B = Tools.remount(Aprime3, (h, w)) #+ 128
	return Aprime2.reshape(h,w), B 

def encodeQuantiseNDecodeBrahimiB(quantization_matrix:ndarray, diagonal_matrix:ndarray):
	quantization_matrix_forward = divide(quantization_matrix, diagonal_matrix)
	quantization_matrix_backward = multiply(diagonal_matrix, quantization_matrix)
	return asarray(quantization_matrix_forward), asarray(quantization_matrix_backward)

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


# MAIN --------------------------------------------------------------------------------------------------
# Pré processamento
path_images = "../../Images_for_tests/Spherical/4K/"
file = "DrivingInCity_3840x1920_30fps_8bit_420_erp_150.bmp"
full_path = os.path.join(path_images, file)
T = calculate_matrix_of_transformation(8)
SO, so = compute_scale_matrix(TO)
SB, sb = compute_scale_matrix(TB)
quantization_factor = range(5, 100, 5)


image = imread(full_path, as_gray=True).astype(float)
if image.max() <= 1:
	image = around(255*image)
h, w = image.shape
A = Tools.umount(image, (8, 8))# - 128
JpegPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T, A), T.T)
OliveiraPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO, A), TO.T)
BrahimiPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB, A), TB.T)

BUFFER = {'JPEG': {'PSNR':[], 'SSIM':[], 'BPP':[]},
			'OLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]}, 
			'BRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]}}

results = []

for QF in quantization_factor:
	QPhiOliveira = prepareQPhi(image, Q0, QF, 8)
	QPhiBrahimi = prepareQPhi(image, QB, QF, 8)
	
	# Jpeg with QPhi
	JpegPrime2 = multiply(around(divide(JpegPrime1, QPhiOliveira)), QPhiOliveira)
	JpegPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, JpegPrime2), T)
	B = clip(Tools.remount(JpegPrime3, (h, w)), 0, 255)
	JpegPrime2 = JpegPrime2.reshape(h, w)
	BUFFER['JPEG']['PSNR'].append(peak_signal_noise_ratio(image, B, data_range=255))
	BUFFER['JPEG']['SSIM'].append(structural_similarity(image, B, data_range=255))
	BUFFER['JPEG']['BPP'].append(bpp(JpegPrime2))

	# Oliveira with QPhi
	ZO = tile(asarray([dot(so.T, so)]), (OliveiraPrime1.shape[0], 1, 1))	# É possível tirar e colocar fora do laço para não fazer o cálculo toda a vez
	QPhiOliveiraForward = np2_round(divide(QPhiOliveira, ZO))
	QPhiOliveiraBackward = np2_round(multiply(QPhiOliveira, ZO))
	OliveiraPrime2 = multiply(around(divide(OliveiraPrime1, QPhiOliveiraForward)), QPhiOliveiraBackward)
	OliveiraPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, OliveiraPrime2), TO)
	C = clip(Tools.remount(OliveiraPrime3, (h, w)), 0, 255)
	OliveiraPrime2 = OliveiraPrime2.reshape(h, w)
	BUFFER['OLIVEIRA']['PSNR'].append(WSPSNR(image, C))
	BUFFER['OLIVEIRA']['SSIM'].append(WSSSIM(image, C))
	BUFFER['OLIVEIRA']['BPP'].append(bpp(OliveiraPrime2))

	# Brahimi with QPhi
	ZB = tile(asarray([dot(sb.T, sb)]), (BrahimiPrime1.shape[0], 1, 1))	# É possível tirar e colocar fora do laço para não fazer o cálculo toda a vez
	QPhiBrahimiForward = np2_ceil(divide(QPhiBrahimi, ZB))
	QPhiBrahimiBackward = np2_ceil(multiply(QPhiBrahimi, ZB))
	BrahimiPrime2 = multiply(around(divide(BrahimiPrime1, QPhiBrahimiForward)), QPhiBrahimiBackward)
	BrahimiPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, BrahimiPrime2), TB)
	D = clip(Tools.remount(BrahimiPrime3, (h, w)), 0, 255)
	BrahimiPrime2 = BrahimiPrime2.reshape(h, w)
	BUFFER['BRAHIMI']['PSNR'].append(WSPSNR(image, D))
	BUFFER['BRAHIMI']['SSIM'].append(WSSSIM(image, D))
	BUFFER['BRAHIMI']['BPP'].append(bpp(BrahimiPrime2))

results.append({'File name':file, 'Method':"JPEG", 'PSNR':BUFFER['JPEG']['PSNR'], 'SSIM':BUFFER['JPEG']['SSIM'], 'BPP':BUFFER['JPEG']['BPP']})
results.append({'File name':file, 'Method':"Oliveira's proposal", 'PSNR':BUFFER['OLIVEIRA']['PSNR'], 'SSIM':BUFFER['OLIVEIRA']['SSIM'], 'BPP':BUFFER['OLIVEIRA']['BPP']})
results.append({'File name':file, 'Method':"Brahimi's proposal", 'PSNR':BUFFER['BRAHIMI']['PSNR'], 'SSIM':BUFFER['BRAHIMI']['SSIM'], 'BPP':BUFFER['BRAHIMI']['BPP']})

results = sorted(results, key=itemgetter('File name'))
fieldnames = ['File name', 'Method', 'PSNR', 'SSIM', 'BPP']
with open('standards.csv', 'w') as csv_file:
	writer = csv.DictWriter(csv_file, fieldnames)
	writer.writeheader()
	for result in results:
		writer.writerow(result)
