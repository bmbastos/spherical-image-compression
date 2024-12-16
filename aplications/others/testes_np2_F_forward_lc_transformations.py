import os
import csv
from numpy import *
from time import time
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize as resize2
from scipy import signal
from matplotlib import pyplot as plot
from pdb import set_trace as pause
from tqdm import tqdm
from operator import itemgetter
from ..mylibs.matrixes import *
from ..mylibs.tools import Tools

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

def adjust_quantization(quality_factor:int, quantization_matrix:ndarray) -> ndarray:
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
""" Função de transformação de uma matriz em uma matriz de potências de dois - Brahimi """

def np2_floor(quantization_matrix:matrix) -> matrix:
	return power(2, floor(log2(quantization_matrix))) # Não usar
""" Função de transformação de uma matriz em uma matriz de potências de dois """

def map_k_and_el(row_index:int, image_height:int) -> tuple:
	el = row_index/image_height * pi - pi/2
	kprime = arange(8)
	k = clip(0, 7, around(kprime/cos(el))).astype(int)
	return (k, el)

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
""" Constroi a Look-up-table da imagem com base em sua altura """


def printLUT(k_lut:ndarray, min_lut:ndarray, max_lut:ndarray):
        for idx in range(len(k_lut)):
            print(k_lut[idx], "%.4f" % min_lut[idx], "%.4f" % max_lut[idx])


def QtildeAtEl(k_lut:ndarray, min_lut:ndarray, max_lut:ndarray, el:float32, quantization_matrix:ndarray, QF:int= 50):
		ks = None
		Q = []
		el = abs(el) # LUT is mirrored
		for idx in range(len(k_lut)):
			if el >= min_lut[idx] and el < max_lut[idx]: 
				ks = k_lut[idx]
		if ks is None and isclose(el, 0): ks = k_lut[0]
		for k in ks:
			Q.append(quantization_matrix.T.tolist()[k])
		Q = asarray(Q)
		Q = Q.T
		return Q


def prepareQPhi(image:ndarray, quantization_matrix:ndarray, N = 8):
	h, w = image.shape
	k_lut, min_lut, max_lut = build_LUT(h)
	els = linspace(-pi/2, pi/2, h//N+1)
	els = 0.5*(els[1:] + els[:-1]) # gets the "central" block elevation
	QPhi = []
	for el in els: 
		QPhi.append(QtildeAtEl(k_lut, min_lut, max_lut, el, quantization_matrix))
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
path_images = os.getcwd() + "/images_for_tests/spherical/by_resolution/4K/"
T = calculate_matrix_of_transformation(8)
SO, so = compute_scale_matrix(TO)
SB, sb = compute_scale_matrix(TB)
SR, sr = compute_scale_matrix(TR)
ZO = dot(so.T, so)
ZB = dot(sb.T, sb)
ZR = dot(sr.T, sr)

quantization_factor = range(5, 100, 5)
results = []
target = 1
processed_images = 0
files = os.listdir(path_images)
for file in files:
	full_path = os.path.join(path_images, file)
	if os.path.isfile(full_path):

		image = imread(full_path, as_gray=True).astype(float)
		image = resize2(image, (512, 1024), anti_aliasing=True)

		if image.max() <= 1:
			image = around(255*image)
		h, w = image.shape
		A = Tools.umount(image, (8, 8))# - 128
		JpegPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T, A), T.T)
		OliveiraPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO, A), TO.T)
		BrahimiPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB, A), TB.T)
		RaizaPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR, A), TR.T)
		ZO_tiled = tile(asarray([ZO]), (A.shape[0], 1, 1))
		ZB_tiled = tile(asarray([ZB]), (A.shape[0], 1, 1))
		ZR_tiled = tile(asarray([ZR]), (A.shape[0], 1, 1))

		BUFFER = {'QJPEG_FC_TOLIVEIRA_TOLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QJPEG_FC_TOLIVEIRA_TBRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QJPEG_FC_TOLIVEIRA_TRAIZA': {'PSNR':[], 'SSIM':[], 'BPP':[]},

					'QJPEG_FF_TBRAHIMI_TOLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QJPEG_FF_TBRAHIMI_TBRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QJPEG_FF_TBRAHIMI_TRAIZA': {'PSNR':[], 'SSIM':[], 'BPP':[]},

					'QJPEG_FR_TBRAHIMI_TOLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QJPEG_FR_TBRAHIMI_TBRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QJPEG_FR_TBRAHIMI_TRAIZA': {'PSNR':[], 'SSIM':[], 'BPP':[]},

					'QBRAHIMI_FC_TOLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QBRAHIMI_FC_TBRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QBRAHIMI_FC_TRAIZA': {'PSNR':[], 'SSIM':[], 'BPP':[]},

					'QBRAHIMI_FF_TOLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QBRAHIMI_FF_TBRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QBRAHIMI_FF_TRAIZA': {'PSNR':[], 'SSIM':[], 'BPP':[]},

					'QBRAHIMI_FR_TOLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QBRAHIMI_FR_TBRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QBRAHIMI_FR_TRAIZA': {'PSNR':[], 'SSIM':[], 'BPP':[]},

					'QHVS_FC_TOLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QHVS_FC_TBRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QHVS_FC_TRAIZA': {'PSNR':[], 'SSIM':[], 'BPP':[]},

					'QHVS_FF_TOLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QHVS_FF_TBRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QHVS_FF_TRAIZA': {'PSNR':[], 'SSIM':[], 'BPP':[]},

					'QHVS_FR_TOLIVEIRA': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QHVS_FR_TBRAHIMI': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QHVS_FR_TRAIZA': {'PSNR':[], 'SSIM':[], 'BPP':[]}}

		for QF in tqdm(quantization_factor):
			''' QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG'''
			''' OLIVEIRA TRANSFORMATIONS -- OLIVEIRA TRANSFORMATIONS -- OLIVEIRA TRANSFORMATIONS -- OLIVEIRA TRANSFORM -- OLIVEIRA TRANSFORMATIONS'''
			QOliveira = adjust_quantization(QF, Q0)
			QPhi_Oliveira_Forward = prepareQPhi(image, np2_floor(divide(QOliveira, ZO)))
			# FLOOR FORWARD + CEIL BACKWARD (FC) + TOLIVEIRA
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_ceil(multiply(QOliveira, ZO)))
			J_FC_OLIVEIRA_Prime2 = multiply(around(divide(OliveiraPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_FC_OLIVEIRA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, J_FC_OLIVEIRA_Prime2), TO)
			J_FC_OLIVEIRA = clip(Tools.remount(J_FC_OLIVEIRA_Prime3, (h, w)), 0, 255)
			J_FC_OLIVEIRA_Prime2 = J_FC_OLIVEIRA_Prime2.reshape(h, w)
			BUFFER['QJPEG_FC_TOLIVEIRA_TOLIVEIRA']['PSNR'].append(WSPSNR(image, J_FC_OLIVEIRA))
			BUFFER['QJPEG_FC_TOLIVEIRA_TOLIVEIRA']['SSIM'].append(WSSSIM(image, J_FC_OLIVEIRA))
			BUFFER['QJPEG_FC_TOLIVEIRA_TOLIVEIRA']['BPP'].append(bpp(J_FC_OLIVEIRA_Prime2))
			del J_FC_OLIVEIRA_Prime2, J_FC_OLIVEIRA_Prime2, J_FC_OLIVEIRA, QPhi_Oliveira_Backward

			# FLOOR FORWARD + FLOOR BACKWARD (FF) + TOLIVEIRA
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_floor(multiply(QOliveira, ZO)))
			J_FF_OLIVEIRA_Prime2 = multiply(around(divide(OliveiraPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_FF_OLIVEIRA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, J_FF_OLIVEIRA_Prime2), TO)
			J_FF_OLIVEIRA = clip(Tools.remount(J_FF_OLIVEIRA_Prime3, (h, w)), 0, 255)
			J_FF_OLIVEIRA_Prime2 = J_FF_OLIVEIRA_Prime2.reshape(h, w)
			BUFFER['QJPEG_FF_TBRAHIMI_TOLIVEIRA']['PSNR'].append(WSPSNR(image, J_FF_OLIVEIRA))
			BUFFER['QJPEG_FF_TBRAHIMI_TOLIVEIRA']['SSIM'].append(WSSSIM(image, J_FF_OLIVEIRA))
			BUFFER['QJPEG_FF_TBRAHIMI_TOLIVEIRA']['BPP'].append(bpp(J_FF_OLIVEIRA_Prime2))
			del J_FF_OLIVEIRA_Prime2, J_FF_OLIVEIRA_Prime3, J_FF_OLIVEIRA, QPhi_Oliveira_Backward

			# FLOOR FORWARD + ROUND BACKWARD (FR) + TOLIVEIRA
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_round(multiply(QOliveira, ZO)))
			J_FR_OLIVEIRA_Prime2 = multiply(around(divide(OliveiraPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_FR_OLIVEIRA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, J_FR_OLIVEIRA_Prime2), TO)
			J_FR_OLIVEIRA = clip(Tools.remount(J_FR_OLIVEIRA_Prime3, (h, w)), 0, 255)
			J_FR_OLIVEIRA_Prime2 = J_FR_OLIVEIRA_Prime2.reshape(h, w)
			BUFFER['QJPEG_FR_TBRAHIMI_TOLIVEIRA']['PSNR'].append(WSPSNR(image, J_FR_OLIVEIRA))
			BUFFER['QJPEG_FR_TBRAHIMI_TOLIVEIRA']['SSIM'].append(WSSSIM(image, J_FR_OLIVEIRA))
			BUFFER['QJPEG_FR_TBRAHIMI_TOLIVEIRA']['BPP'].append(bpp(J_FR_OLIVEIRA_Prime2))
			del J_FR_OLIVEIRA_Prime2, J_FR_OLIVEIRA_Prime3, J_FR_OLIVEIRA, QPhi_Oliveira_Backward, QPhi_Oliveira_Forward

			''' TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI '''
			QPhi_Oliveira_Forward = prepareQPhi(image, np2_floor(divide(QOliveira, ZB)))
			# FLOOR FORWARD + CEIL BACKWARD (FC) + TBRAHIMI
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_ceil(multiply(QOliveira, ZB)))
			J_FC_BRAHIMI_Prime2 = multiply(around(divide(BrahimiPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_FC_BRAHIMI_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, J_FC_BRAHIMI_Prime2), TB)
			J_FC_BRAHIMI = clip(Tools.remount(J_FC_BRAHIMI_Prime3, (h, w)), 0, 255)
			J_FC_BRAHIMI_Prime2 = J_FC_BRAHIMI_Prime2.reshape(h, w)
			BUFFER['QJPEG_FC_TBRAHIMI']['PSNR'].append(WSPSNR(image, J_FC_BRAHIMI))
			BUFFER['QJPEG_FC_TBRAHIMI']['SSIM'].append(WSSSIM(image, J_FC_BRAHIMI))
			BUFFER['QJPEG_FC_TBRAHIMI']['BPP'].append(bpp(J_FC_BRAHIMI_Prime2))
			del J_FC_BRAHIMI_Prime2, J_FC_BRAHIMI_Prime2, J_FC_BRAHIMI, QPhi_Oliveira_Backward

			# FLOOR FORWARD + FLOOR BACKWARD (FF) + TBRAHIMI
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_floor(multiply(QOliveira, ZB)))
			J_FF_BRAHIMI_Prime2 = multiply(around(divide(BrahimiPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_FF_BRAHIMI_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, J_FF_BRAHIMI_Prime2), TB)
			J_FF_BRAHIMI = clip(Tools.remount(J_FF_BRAHIMI_Prime3, (h, w)), 0, 255)
			J_FF_BRAHIMI_Prime2 = J_FF_BRAHIMI_Prime2.reshape(h, w)
			BUFFER['QJPEG_FF_TBRAHIMI']['PSNR'].append(WSPSNR(image, J_FF_BRAHIMI))
			BUFFER['QJPEG_FF_TBRAHIMI']['SSIM'].append(WSSSIM(image, J_FF_BRAHIMI))
			BUFFER['QJPEG_FF_TBRAHIMI']['BPP'].append(bpp(J_FF_BRAHIMI_Prime2))
			del J_FF_BRAHIMI_Prime2, J_FF_BRAHIMI_Prime3, J_FF_BRAHIMI, QPhi_Oliveira_Backward

			# FLOOR FORWARD + ROUND BACKWARD (FR) + TBRAHIMI
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_round(multiply(QOliveira, ZB)))
			J_FR_BRAHIMI_Prime2 = multiply(around(divide(BrahimiPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_FR_BRAHIMI_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, J_FR_BRAHIMI_Prime2), TB)
			J_FR_BRAHIMI = clip(Tools.remount(J_FR_BRAHIMI_Prime3, (h, w)), 0, 255)
			J_FR_BRAHIMI_Prime2 = J_FR_BRAHIMI_Prime2.reshape(h, w)
			BUFFER['QJPEG_FR_TBRAHIMI']['PSNR'].append(WSPSNR(image, J_FR_BRAHIMI))
			BUFFER['QJPEG_FR_TBRAHIMI']['SSIM'].append(WSSSIM(image, J_FR_BRAHIMI))
			BUFFER['QJPEG_FR_TBRAHIMI']['BPP'].append(bpp(J_FR_BRAHIMI_Prime2))
			del J_FR_BRAHIMI_Prime2, J_FR_BRAHIMI_Prime3, J_FR_BRAHIMI, QPhi_Oliveira_Backward, QPhi_Oliveira_Forward

			''' TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- '''
			QPhi_Oliveira_Forward = prepareQPhi(image, np2_floor(divide(QOliveira, ZR)))
			# FLOOR FORWARD + CEIL BACKWARD (FC) + TRAIZA
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_ceil(multiply(QOliveira, ZR)))
			J_FC_RAIZA_Prime2 = multiply(around(divide(RaizaPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_FC_RAIZA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR.T, J_FC_RAIZA_Prime2), TR)
			J_FC_RAIZA = clip(Tools.remount(J_FC_RAIZA_Prime3, (h, w)), 0, 255)
			J_FC_RAIZA_Prime2 = J_FC_RAIZA_Prime2.reshape(h, w)
			BUFFER['QJPEG_FC_TRAIZA']['PSNR'].append(WSPSNR(image, J_FC_RAIZA))
			BUFFER['QJPEG_FC_TRAIZA']['SSIM'].append(WSSSIM(image, J_FC_RAIZA))
			BUFFER['QJPEG_FC_TRAIZA']['BPP'].append(bpp(J_FC_RAIZA_Prime2))
			del J_FC_RAIZA_Prime2, J_FC_RAIZA_Prime2, J_FC_RAIZA, QPhi_Oliveira_Backward

			# FLOOR FORWARD + FLOOR BACKWARD (FF) + TRAIZA
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_floor(multiply(QOliveira, ZR)))
			J_FF_RAIZA_Prime2 = multiply(around(divide(RaizaPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_FF_RAIZA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR.T, J_FF_RAIZA_Prime2), TR)
			J_FF_RAIZA = clip(Tools.remount(J_FF_RAIZA_Prime3, (h, w)), 0, 255)
			J_FF_RAIZA_Prime2 = J_FF_RAIZA_Prime2.reshape(h, w)
			BUFFER['QJPEG_FF_TRAIZA']['PSNR'].append(WSPSNR(image, J_FF_RAIZA))
			BUFFER['QJPEG_FF_TRAIZA']['SSIM'].append(WSSSIM(image, J_FF_RAIZA))
			BUFFER['QJPEG_FF_TRAIZA']['BPP'].append(bpp(J_FF_RAIZA_Prime2))
			del J_FF_RAIZA_Prime2, J_FF_RAIZA_Prime3, J_FF_RAIZA, QPhi_Oliveira_Backward

			# FLOOR FORWARD + ROUND BACKWARD (FR) + TRAIZA
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_round(multiply(QOliveira, ZR)))
			J_FR_RAIZA_Prime2 = multiply(around(divide(RaizaPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_FR_RAIZA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR.T, J_FR_RAIZA_Prime2), TR)
			J_FR_RAIZA = clip(Tools.remount(J_FR_RAIZA_Prime3, (h, w)), 0, 255)
			J_FR_RAIZA_Prime2 = J_FR_RAIZA_Prime2.reshape(h, w)
			BUFFER['QJPEG_FR_TRAIZA']['PSNR'].append(WSPSNR(image, J_FR_RAIZA))
			BUFFER['QJPEG_FR_TRAIZA']['SSIM'].append(WSSSIM(image, J_FR_RAIZA))
			BUFFER['QJPEG_FR_TRAIZA']['BPP'].append(bpp(J_FR_RAIZA_Prime2))
			del J_FR_RAIZA_Prime2, J_FR_RAIZA_Prime3, J_FR_RAIZA, QPhi_Oliveira_Backward, QPhi_Oliveira_Forward
			



			''' QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI'''
			''' OLIVEIRA TRANSFORMATIONS -- OLIVEIRA TRANSFORMATIONS -- OLIVEIRA TRANSFORMATIONS -- OLIVEIRA TRANSFORM -- OLIVEIRA TRANSFORMATIONS'''
			QBrahimi = adjust_quantization(QF, QB)
			QPhi_Brahimi_Forward = prepareQPhi(image, np2_floor(divide(QBrahimi, ZO)))
			# FLOOR FORWARD + CEIL BACKWARD (FC) + TOLIVEIRA
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_ceil(multiply(QBrahimi, ZO)))
			B_FC_OLIVEIRA_Prime2 = multiply(around(divide(OliveiraPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_FC_OLIVEIRA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, B_FC_OLIVEIRA_Prime2), TO)
			B_FC_OLIVEIRA = clip(Tools.remount(B_FC_OLIVEIRA_Prime3, (h, w)), 0, 255)
			B_FC_OLIVEIRA_Prime2 = B_FC_OLIVEIRA_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_FC_TBRAHIMI_TOLIVEIRA']['PSNR'].append(WSPSNR(image, B_FC_OLIVEIRA))
			BUFFER['QBRAHIMI_FC_TBRAHIMI_TOLIVEIRA']['SSIM'].append(WSSSIM(image, B_FC_OLIVEIRA))
			BUFFER['QBRAHIMI_FC_TBRAHIMI_TOLIVEIRA']['BPP'].append(bpp(B_FC_OLIVEIRA_Prime2))
			del B_FC_OLIVEIRA_Prime2, B_FC_OLIVEIRA_Prime2, B_FC_OLIVEIRA, QPhi_Brahimi_Backward

			# FLOOR FORWARD + FLOOR BACKWARD (FF) + TOLIVEIRA
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_floor(multiply(QBrahimi, ZO)))
			B_FF_OLIVEIRA_Prime2 = multiply(around(divide(OliveiraPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_FF_OLIVEIRA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, B_FF_OLIVEIRA_Prime2), TO)
			B_FF_OLIVEIRA = clip(Tools.remount(B_FF_OLIVEIRA_Prime3, (h, w)), 0, 255)
			B_FF_OLIVEIRA_Prime2 = B_FF_OLIVEIRA_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_FF_TBRAHIMI_TOLIVEIRA']['PSNR'].append(WSPSNR(image, B_FF_OLIVEIRA))
			BUFFER['QBRAHIMI_FF_TBRAHIMI_TOLIVEIRA']['SSIM'].append(WSSSIM(image, B_FF_OLIVEIRA))
			BUFFER['QBRAHIMI_FF_TBRAHIMI_TOLIVEIRA']['BPP'].append(bpp(B_FF_OLIVEIRA_Prime2))
			del B_FF_OLIVEIRA_Prime2, B_FF_OLIVEIRA_Prime3, B_FF_OLIVEIRA, QPhi_Brahimi_Backward

			# FLOOR FORWARD + ROUND BACKWARD (FR) + TOLIVEIRA
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_round(multiply(QBrahimi, ZO)))
			B_FR_OLIVEIRA_Prime2 = multiply(around(divide(OliveiraPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_FR_OLIVEIRA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, B_FR_OLIVEIRA_Prime2), TO)
			B_FR_OLIVEIRA = clip(Tools.remount(B_FR_OLIVEIRA_Prime3, (h, w)), 0, 255)
			B_FR_OLIVEIRA_Prime2 = B_FR_OLIVEIRA_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_FR_TBRAHIMI_TOLIVEIRA']['PSNR'].append(WSPSNR(image, B_FR_OLIVEIRA))
			BUFFER['QBRAHIMI_FR_TBRAHIMI_TOLIVEIRA']['SSIM'].append(WSSSIM(image, B_FR_OLIVEIRA))
			BUFFER['QBRAHIMI_FR_TBRAHIMI_TOLIVEIRA']['BPP'].append(bpp(B_FR_OLIVEIRA_Prime2))
			del B_FR_OLIVEIRA_Prime2, B_FR_OLIVEIRA_Prime3, B_FR_OLIVEIRA, QPhi_Brahimi_Backward, QPhi_Brahimi_Forward

			''' TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI '''
			QPhi_Brahimi_Forward = prepareQPhi(image, np2_floor(divide(QBrahimi, ZB)))
			# FLOOR FORWARD + CEIL BACKWARD (FC) + TBRAHIMI
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_ceil(multiply(QBrahimi, ZB)))
			B_FC_BRAHIMI_Prime2 = multiply(around(divide(BrahimiPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_FC_BRAHIMI_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, B_FC_BRAHIMI_Prime2), TB)
			B_FC_BRAHIMI = clip(Tools.remount(B_FC_BRAHIMI_Prime3, (h, w)), 0, 255)
			B_FC_BRAHIMI_Prime2 = B_FC_BRAHIMI_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_FC_TBRAHIMI']['PSNR'].append(WSPSNR(image, B_FC_BRAHIMI))
			BUFFER['QBRAHIMI_FC_TBRAHIMI']['SSIM'].append(WSSSIM(image, B_FC_BRAHIMI))
			BUFFER['QBRAHIMI_FC_TBRAHIMI']['BPP'].append(bpp(B_FC_BRAHIMI_Prime2))
			del B_FC_BRAHIMI_Prime2, B_FC_BRAHIMI_Prime2, B_FC_BRAHIMI, QPhi_Brahimi_Backward

			# FLOOR FORWARD + FLOOR BACKWARD (FF) + TBRAHIMI
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_floor(multiply(QBrahimi, ZB)))
			B_FF_BRAHIMI_Prime2 = multiply(around(divide(BrahimiPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_FF_BRAHIMI_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, B_FF_BRAHIMI_Prime2), TB)
			B_FF_BRAHIMI = clip(Tools.remount(B_FF_BRAHIMI_Prime3, (h, w)), 0, 255)
			B_FF_BRAHIMI_Prime2 = B_FF_BRAHIMI_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_FF_TBRAHIMI']['PSNR'].append(WSPSNR(image, B_FF_BRAHIMI))
			BUFFER['QBRAHIMI_FF_TBRAHIMI']['SSIM'].append(WSSSIM(image, B_FF_BRAHIMI))
			BUFFER['QBRAHIMI_FF_TBRAHIMI']['BPP'].append(bpp(B_FF_BRAHIMI_Prime2))
			del B_FF_BRAHIMI_Prime2, B_FF_BRAHIMI_Prime3, B_FF_BRAHIMI, QPhi_Brahimi_Backward

			# FLOOR FORWARD + ROUND BACKWARD (FR) + TBRAHIMI
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_round(multiply(QBrahimi, ZB)))
			B_FR_BRAHIMI_Prime2 = multiply(around(divide(BrahimiPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_FR_BRAHIMI_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, B_FR_BRAHIMI_Prime2), TB)
			B_FR_BRAHIMI = clip(Tools.remount(B_FR_BRAHIMI_Prime3, (h, w)), 0, 255)
			B_FR_BRAHIMI_Prime2 = B_FR_BRAHIMI_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_FR_TBRAHIMI']['PSNR'].append(WSPSNR(image, B_FR_BRAHIMI))
			BUFFER['QBRAHIMI_FR_TBRAHIMI']['SSIM'].append(WSSSIM(image, B_FR_BRAHIMI))
			BUFFER['QBRAHIMI_FR_TBRAHIMI']['BPP'].append(bpp(B_FR_BRAHIMI_Prime2))
			del B_FR_BRAHIMI_Prime2, B_FR_BRAHIMI_Prime3, B_FR_BRAHIMI, QPhi_Brahimi_Backward, QPhi_Brahimi_Forward

			''' TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- '''
			QPhi_Brahimi_Forward = prepareQPhi(image, np2_floor(divide(QBrahimi, ZR)))
			# FLOOR FORWARD + CEIL BACKWARD (FC) + TRAIZA
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_ceil(multiply(QBrahimi, ZR)))
			B_FC_RAIZA_Prime2 = multiply(around(divide(RaizaPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_FC_RAIZA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR.T, B_FC_RAIZA_Prime2), TR)
			B_FC_RAIZA = clip(Tools.remount(B_FC_RAIZA_Prime3, (h, w)), 0, 255)
			B_FC_RAIZA_Prime2 = B_FC_RAIZA_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_FC_TRAIZA']['PSNR'].append(WSPSNR(image, B_FC_RAIZA))
			BUFFER['QBRAHIMI_FC_TRAIZA']['SSIM'].append(WSSSIM(image, B_FC_RAIZA))
			BUFFER['QBRAHIMI_FC_TRAIZA']['BPP'].append(bpp(B_FC_RAIZA_Prime2))
			del B_FC_RAIZA_Prime2, B_FC_RAIZA_Prime2, B_FC_RAIZA, QPhi_Brahimi_Backward

			# FLOOR FORWARD + FLOOR BACKWARD (FF) + TRAIZA
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_floor(multiply(QBrahimi, ZR)))
			B_FF_RAIZA_Prime2 = multiply(around(divide(RaizaPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_FF_RAIZA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR.T, B_FF_RAIZA_Prime2), TR)
			B_FF_RAIZA = clip(Tools.remount(B_FF_RAIZA_Prime3, (h, w)), 0, 255)
			B_FF_RAIZA_Prime2 = B_FF_RAIZA_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_FF_TRAIZA']['PSNR'].append(WSPSNR(image, B_FF_RAIZA))
			BUFFER['QBRAHIMI_FF_TRAIZA']['SSIM'].append(WSSSIM(image, B_FF_RAIZA))
			BUFFER['QBRAHIMI_FF_TRAIZA']['BPP'].append(bpp(B_FF_RAIZA_Prime2))
			del B_FF_RAIZA_Prime2, B_FF_RAIZA_Prime3, B_FF_RAIZA, QPhi_Brahimi_Backward

			# FLOOR FORWARD + ROUND BACKWARD (FR) + TRAIZA
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_round(multiply(QBrahimi, ZR)))
			B_FR_RAIZA_Prime2 = multiply(around(divide(RaizaPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_FR_RAIZA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR.T, B_FR_RAIZA_Prime2), TR)
			B_FR_RAIZA = clip(Tools.remount(B_FR_RAIZA_Prime3, (h, w)), 0, 255)
			B_FR_RAIZA_Prime2 = B_FR_RAIZA_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_FR_TRAIZA']['PSNR'].append(WSPSNR(image, B_FR_RAIZA))
			BUFFER['QBRAHIMI_FR_TRAIZA']['SSIM'].append(WSSSIM(image, B_FR_RAIZA))
			BUFFER['QBRAHIMI_FR_TRAIZA']['BPP'].append(bpp(B_FR_RAIZA_Prime2))
			del B_FR_RAIZA_Prime2, B_FR_RAIZA_Prime3, B_FR_RAIZA, QPhi_Brahimi_Backward, QPhi_Brahimi_Forward

			





			''' QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS'''
			''' OLIVEIRA TRANSFORMATIONS -- OLIVEIRA TRANSFORMATIONS -- OLIVEIRA TRANSFORMATIONS -- OLIVEIRA TRANSFORM -- OLIVEIRA TRANSFORMATIONS'''
			QHvs = adjust_quantization(QF, Qhvs)
			QPhi_QHvs_Forward = prepareQPhi(image, np2_floor(divide(QHvs, ZO)))
			# FLOOR FORWARD + CEIL BACKWARD (FC) + TOLIVEIRA
			QPhi_QHvs_Backward = prepareQPhi(image, np2_ceil(multiply(QHvs, ZO)))
			B_FC_OLIVEIRA_Prime2 = multiply(around(divide(OliveiraPrime1, QPhi_QHvs_Forward)), QPhi_QHvs_Backward)
			B_FC_OLIVEIRA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, B_FC_OLIVEIRA_Prime2), TO)
			B_FC_OLIVEIRA = clip(Tools.remount(B_FC_OLIVEIRA_Prime3, (h, w)), 0, 255)
			B_FC_OLIVEIRA_Prime2 = B_FC_OLIVEIRA_Prime2.reshape(h, w)
			BUFFER['QHVS_FC_TBRAHIMI_TOLIVEIRA']['PSNR'].append(WSPSNR(image, B_FC_OLIVEIRA))
			BUFFER['QHVS_FC_TBRAHIMI_TOLIVEIRA']['SSIM'].append(WSSSIM(image, B_FC_OLIVEIRA))
			BUFFER['QHVS_FC_TBRAHIMI_TOLIVEIRA']['BPP'].append(bpp(B_FC_OLIVEIRA_Prime2))
			del B_FC_OLIVEIRA_Prime2, B_FC_OLIVEIRA_Prime2, B_FC_OLIVEIRA, QPhi_QHvs_Backward

			# FLOOR FORWARD + FLOOR BACKWARD (FF) + TOLIVEIRA
			QPhi_QHvs_Backward = prepareQPhi(image, np2_floor(multiply(QHvs, ZO)))
			B_FF_OLIVEIRA_Prime2 = multiply(around(divide(OliveiraPrime1, QPhi_QHvs_Forward)), QPhi_QHvs_Backward)
			B_FF_OLIVEIRA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, B_FF_OLIVEIRA_Prime2), TO)
			B_FF_OLIVEIRA = clip(Tools.remount(B_FF_OLIVEIRA_Prime3, (h, w)), 0, 255)
			B_FF_OLIVEIRA_Prime2 = B_FF_OLIVEIRA_Prime2.reshape(h, w)
			BUFFER['QHVS_FF_TBRAHIMI_TOLIVEIRA']['PSNR'].append(WSPSNR(image, B_FF_OLIVEIRA))
			BUFFER['QHVS_FF_TBRAHIMI_TOLIVEIRA']['SSIM'].append(WSSSIM(image, B_FF_OLIVEIRA))
			BUFFER['QHVS_FF_TBRAHIMI_TOLIVEIRA']['BPP'].append(bpp(B_FF_OLIVEIRA_Prime2))
			del B_FF_OLIVEIRA_Prime2, B_FF_OLIVEIRA_Prime3, B_FF_OLIVEIRA, QPhi_QHvs_Backward

			# FLOOR FORWARD + ROUND BACKWARD (FR) + TOLIVEIRA
			QPhi_QHvs_Backward = prepareQPhi(image, np2_round(multiply(QHvs, ZO)))
			B_FR_OLIVEIRA_Prime2 = multiply(around(divide(OliveiraPrime1, QPhi_QHvs_Forward)), QPhi_QHvs_Backward)
			B_FR_OLIVEIRA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, B_FR_OLIVEIRA_Prime2), TO)
			B_FR_OLIVEIRA = clip(Tools.remount(B_FR_OLIVEIRA_Prime3, (h, w)), 0, 255)
			B_FR_OLIVEIRA_Prime2 = B_FR_OLIVEIRA_Prime2.reshape(h, w)
			BUFFER['QHVS_FR_TBRAHIMI_TOLIVEIRA']['PSNR'].append(WSPSNR(image, B_FR_OLIVEIRA))
			BUFFER['QHVS_FR_TBRAHIMI_TOLIVEIRA']['SSIM'].append(WSSSIM(image, B_FR_OLIVEIRA))
			BUFFER['QHVS_FR_TBRAHIMI_TOLIVEIRA']['BPP'].append(bpp(B_FR_OLIVEIRA_Prime2))
			del B_FR_OLIVEIRA_Prime2, B_FR_OLIVEIRA_Prime3, B_FR_OLIVEIRA, QPhi_QHvs_Backward, QPhi_QHvs_Forward

			''' TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI -- TBRAHIMI '''
			QPhi_QHvs_Forward = prepareQPhi(image, np2_floor(divide(QHvs, ZB)))
			# FLOOR FORWARD + CEIL BACKWARD (FC) + TBRAHIMI
			QPhi_QHvs_Backward = prepareQPhi(image, np2_ceil(multiply(QHvs, ZB)))
			B_FC_BRAHIMI_Prime2 = multiply(around(divide(BrahimiPrime1, QPhi_QHvs_Forward)), QPhi_QHvs_Backward)
			B_FC_BRAHIMI_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, B_FC_BRAHIMI_Prime2), TB)
			B_FC_BRAHIMI = clip(Tools.remount(B_FC_BRAHIMI_Prime3, (h, w)), 0, 255)
			B_FC_BRAHIMI_Prime2 = B_FC_BRAHIMI_Prime2.reshape(h, w)
			BUFFER['QHVS_FC_TBRAHIMI']['PSNR'].append(WSPSNR(image, B_FC_BRAHIMI))
			BUFFER['QHVS_FC_TBRAHIMI']['SSIM'].append(WSSSIM(image, B_FC_BRAHIMI))
			BUFFER['QHVS_FC_TBRAHIMI']['BPP'].append(bpp(B_FC_BRAHIMI_Prime2))
			del B_FC_BRAHIMI_Prime2, B_FC_BRAHIMI_Prime2, B_FC_BRAHIMI, QPhi_QHvs_Backward

			# FLOOR FORWARD + FLOOR BACKWARD (FF) + TBRAHIMI
			QPhi_QHvs_Backward = prepareQPhi(image, np2_floor(multiply(QHvs, ZB)))
			B_FF_BRAHIMI_Prime2 = multiply(around(divide(BrahimiPrime1, QPhi_QHvs_Forward)), QPhi_QHvs_Backward)
			B_FF_BRAHIMI_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, B_FF_BRAHIMI_Prime2), TB)
			B_FF_BRAHIMI = clip(Tools.remount(B_FF_BRAHIMI_Prime3, (h, w)), 0, 255)
			B_FF_BRAHIMI_Prime2 = B_FF_BRAHIMI_Prime2.reshape(h, w)
			BUFFER['QHVS_FF_TBRAHIMI']['PSNR'].append(WSPSNR(image, B_FF_BRAHIMI))
			BUFFER['QHVS_FF_TBRAHIMI']['SSIM'].append(WSSSIM(image, B_FF_BRAHIMI))
			BUFFER['QHVS_FF_TBRAHIMI']['BPP'].append(bpp(B_FF_BRAHIMI_Prime2))
			del B_FF_BRAHIMI_Prime2, B_FF_BRAHIMI_Prime3, B_FF_BRAHIMI, QPhi_QHvs_Backward

			# FLOOR FORWARD + ROUND BACKWARD (FR) + TBRAHIMI
			QPhi_QHvs_Backward = prepareQPhi(image, np2_round(multiply(QHvs, ZB)))
			B_FR_BRAHIMI_Prime2 = multiply(around(divide(BrahimiPrime1, QPhi_QHvs_Forward)), QPhi_QHvs_Backward)
			B_FR_BRAHIMI_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, B_FR_BRAHIMI_Prime2), TB)
			B_FR_BRAHIMI = clip(Tools.remount(B_FR_BRAHIMI_Prime3, (h, w)), 0, 255)
			B_FR_BRAHIMI_Prime2 = B_FR_BRAHIMI_Prime2.reshape(h, w)
			BUFFER['QHVS_FR_TBRAHIMI']['PSNR'].append(WSPSNR(image, B_FR_BRAHIMI))
			BUFFER['QHVS_FR_TBRAHIMI']['SSIM'].append(WSSSIM(image, B_FR_BRAHIMI))
			BUFFER['QHVS_FR_TBRAHIMI']['BPP'].append(bpp(B_FR_BRAHIMI_Prime2))
			del B_FR_BRAHIMI_Prime2, B_FR_BRAHIMI_Prime3, B_FR_BRAHIMI, QPhi_QHvs_Backward, QPhi_QHvs_Forward

			''' TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- TRAIZA -- '''
			QPhi_QHvs_Forward = prepareQPhi(image, np2_floor(divide(QHvs, ZR)))
			# FLOOR FORWARD + CEIL BACKWARD (FC) + TRAIZA
			QPhi_QHvs_Backward = prepareQPhi(image, np2_ceil(multiply(QHvs, ZR)))
			B_FC_RAIZA_Prime2 = multiply(around(divide(RaizaPrime1, QPhi_QHvs_Forward)), QPhi_QHvs_Backward)
			B_FC_RAIZA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR.T, B_FC_RAIZA_Prime2), TR)
			B_FC_RAIZA = clip(Tools.remount(B_FC_RAIZA_Prime3, (h, w)), 0, 255)
			B_FC_RAIZA_Prime2 = B_FC_RAIZA_Prime2.reshape(h, w)
			BUFFER['QHVS_FC_TRAIZA']['PSNR'].append(WSPSNR(image, B_FC_RAIZA))
			BUFFER['QHVS_FC_TRAIZA']['SSIM'].append(WSSSIM(image, B_FC_RAIZA))
			BUFFER['QHVS_FC_TRAIZA']['BPP'].append(bpp(B_FC_RAIZA_Prime2))
			del B_FC_RAIZA_Prime2, B_FC_RAIZA_Prime2, B_FC_RAIZA, QPhi_QHvs_Backward

			# FLOOR FORWARD + FLOOR BACKWARD (FF) + TRAIZA
			QPhi_QHvs_Backward = prepareQPhi(image, np2_floor(multiply(QHvs, ZR)))
			B_FF_RAIZA_Prime2 = multiply(around(divide(RaizaPrime1, QPhi_QHvs_Forward)), QPhi_QHvs_Backward)
			B_FF_RAIZA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR.T, B_FF_RAIZA_Prime2), TR)
			B_FF_RAIZA = clip(Tools.remount(B_FF_RAIZA_Prime3, (h, w)), 0, 255)
			B_FF_RAIZA_Prime2 = B_FF_RAIZA_Prime2.reshape(h, w)
			BUFFER['QHVS_FF_TRAIZA']['PSNR'].append(WSPSNR(image, B_FF_RAIZA))
			BUFFER['QHVS_FF_TRAIZA']['SSIM'].append(WSSSIM(image, B_FF_RAIZA))
			BUFFER['QHVS_FF_TRAIZA']['BPP'].append(bpp(B_FF_RAIZA_Prime2))
			del B_FF_RAIZA_Prime2, B_FF_RAIZA_Prime3, B_FF_RAIZA, QPhi_QHvs_Backward

			# FLOOR FORWARD + ROUND BACKWARD (FR) + TRAIZA
			QPhi_QHvs_Backward = prepareQPhi(image, np2_round(multiply(QHvs, ZR)))
			B_FR_RAIZA_Prime2 = multiply(around(divide(RaizaPrime1, QPhi_QHvs_Forward)), QPhi_QHvs_Backward)
			B_FR_RAIZA_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR.T, B_FR_RAIZA_Prime2), TR)
			B_FR_RAIZA = clip(Tools.remount(B_FR_RAIZA_Prime3, (h, w)), 0, 255)
			B_FR_RAIZA_Prime2 = B_FR_RAIZA_Prime2.reshape(h, w)
			BUFFER['QHVS_FR_TRAIZA']['PSNR'].append(WSPSNR(image, B_FR_RAIZA))
			BUFFER['QHVS_FR_TRAIZA']['SSIM'].append(WSSSIM(image, B_FR_RAIZA))
			BUFFER['QHVS_FR_TRAIZA']['BPP'].append(bpp(B_FR_RAIZA_Prime2))
			del B_FR_RAIZA_Prime2, B_FR_RAIZA_Prime3, B_FR_RAIZA, QPhi_QHvs_Backward, QPhi_QHvs_Forward



			
		processed_images += 1
		results.append({'File name':file, 'Method':"JPEG_FC_TOLIVEIRA", 'PSNR':BUFFER['QJPEG_FC_TOLIVEIRA']['PSNR'], 'SSIM':BUFFER['QJPEG_FC_TOLIVEIRA']['SSIM'], 'BPP':BUFFER['QJPEG_FC_TOLIVEIRA']['BPP']})
		results.append({'File name':file, 'Method':"JPEG_FF_TOLIVEIRA", 'PSNR':BUFFER['QJPEG_FF_TOLIVEIRA']['PSNR'], 'SSIM':BUFFER['QJPEG_FF_TOLIVEIRA']['SSIM'], 'BPP':BUFFER['QJPEG_FF_TOLIVEIRA']['BPP']})
		results.append({'File name':file, 'Method':"JPEG_FR_TOLIVEIRA", 'PSNR':BUFFER['QJPEG_FR_TOLIVEIRA']['PSNR'], 'SSIM':BUFFER['QJPEG_FR_TOLIVEIRA']['SSIM'], 'BPP':BUFFER['QJPEG_FR_TOLIVEIRA']['BPP']})
		results.append({'File name':file, 'Method':"JPEG_FC_TBRAHIMI", 'PSNR':BUFFER['QJPEG_FC_TBRAHIMI']['PSNR'], 'SSIM':BUFFER['QJPEG_FC_TBRAHIMI']['SSIM'], 'BPP':BUFFER['QJPEG_FC_TBRAHIMI']['BPP']})
		results.append({'File name':file, 'Method':"JPEG_FF_TBRAHIMI", 'PSNR':BUFFER['QJPEG_FF_TBRAHIMI']['PSNR'], 'SSIM':BUFFER['QJPEG_FF_TBRAHIMI']['SSIM'], 'BPP':BUFFER['QJPEG_FF_TBRAHIMI']['BPP']})
		results.append({'File name':file, 'Method':"JPEG_FR_TBRAHIMI", 'PSNR':BUFFER['QJPEG_FR_TBRAHIMI']['PSNR'], 'SSIM':BUFFER['QJPEG_FR_TBRAHIMI']['SSIM'], 'BPP':BUFFER['QJPEG_FR_TBRAHIMI']['BPP']})
		results.append({'File name':file, 'Method':"JPEG_FC_TRAIZA", 'PSNR':BUFFER['QJPEG_FC_TRAIZA']['PSNR'], 'SSIM':BUFFER['QJPEG_FC_TRAIZA']['SSIM'], 'BPP':BUFFER['QJPEG_FC_TRAIZA']['BPP']})
		results.append({'File name':file, 'Method':"JPEG_FF_TRAIZA", 'PSNR':BUFFER['QJPEG_FF_TRAIZA']['PSNR'], 'SSIM':BUFFER['QJPEG_FF_TRAIZA']['SSIM'], 'BPP':BUFFER['QJPEG_FF_TRAIZA']['BPP']})
		results.append({'File name':file, 'Method':"JPEG_FR_TRAIZA", 'PSNR':BUFFER['QJPEG_FR_TRAIZA']['PSNR'], 'SSIM':BUFFER['QJPEG_FR_TRAIZA']['SSIM'], 'BPP':BUFFER['QJPEG_FR_TRAIZA']['BPP']})
		results.append({'File name':file, 'Method':"QBRAHIMI_FC_TOLIVEIRA", 'PSNR':BUFFER['QBRAHIMI_FC_TOLIVEIRA']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_FC_TOLIVEIRA']['SSIM'], 'BPP':BUFFER['QBRAHIMI_FC_TOLIVEIRA']['BPP']})
		results.append({'File name':file, 'Method':"QBRAHIMI_FF_TOLIVEIRA", 'PSNR':BUFFER['QBRAHIMI_FF_TOLIVEIRA']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_FF_TOLIVEIRA']['SSIM'], 'BPP':BUFFER['QBRAHIMI_FF_TOLIVEIRA']['BPP']})
		results.append({'File name':file, 'Method':"QBRAHIMI_FR_TOLIVEIRA", 'PSNR':BUFFER['QBRAHIMI_FR_TOLIVEIRA']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_FR_TOLIVEIRA']['SSIM'], 'BPP':BUFFER['QBRAHIMI_FR_TOLIVEIRA']['BPP']})
		results.append({'File name':file, 'Method':"QBRAHIMI_FC_TBRAHIMI", 'PSNR':BUFFER['QBRAHIMI_FC_TBRAHIMI']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_FC_TBRAHIMI']['SSIM'], 'BPP':BUFFER['QBRAHIMI_FC_TBRAHIMI']['BPP']})
		results.append({'File name':file, 'Method':"QBRAHIMI_FF_TBRAHIMI", 'PSNR':BUFFER['QBRAHIMI_FF_TBRAHIMI']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_FF_TBRAHIMI']['SSIM'], 'BPP':BUFFER['QBRAHIMI_FF_TBRAHIMI']['BPP']})
		results.append({'File name':file, 'Method':"QBRAHIMI_FR_TBRAHIMI", 'PSNR':BUFFER['QBRAHIMI_FR_TBRAHIMI']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_FR_TBRAHIMI']['SSIM'], 'BPP':BUFFER['QBRAHIMI_FR_TBRAHIMI']['BPP']})
		results.append({'File name':file, 'Method':"QBRAHIMI_FC_TRAIZA", 'PSNR':BUFFER['QBRAHIMI_FC_TRAIZA']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_FC_TRAIZA']['SSIM'], 'BPP':BUFFER['QBRAHIMI_FC_TRAIZA']['BPP']})
		results.append({'File name':file, 'Method':"QBRAHIMI_FF_TRAIZA", 'PSNR':BUFFER['QBRAHIMI_FF_TRAIZA']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_FF_TRAIZA']['SSIM'], 'BPP':BUFFER['QBRAHIMI_FF_TRAIZA']['BPP']})
		results.append({'File name':file, 'Method':"QBRAHIMI_FR_TRAIZA", 'PSNR':BUFFER['QBRAHIMI_FR_TRAIZA']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_FR_TRAIZA']['SSIM'], 'BPP':BUFFER['QBRAHIMI_FR_TRAIZA']['BPP']})
		results.append({'File name':file, 'Method':"QHVS_FC_TOLIVEIRA", 'PSNR':BUFFER['QHVS_FC_TOLIVEIRA']['PSNR'], 'SSIM':BUFFER['QHVS_FC_TOLIVEIRA']['SSIM'], 'BPP':BUFFER['QHVS_FC_TOLIVEIRA']['BPP']})
		results.append({'File name':file, 'Method':"QHVS_FF_TOLIVEIRA", 'PSNR':BUFFER['QHVS_FF_TOLIVEIRA']['PSNR'], 'SSIM':BUFFER['QHVS_FF_TOLIVEIRA']['SSIM'], 'BPP':BUFFER['QHVS_FF_TOLIVEIRA']['BPP']})
		results.append({'File name':file, 'Method':"QHVS_FR_TOLIVEIRA", 'PSNR':BUFFER['QHVS_FR_TOLIVEIRA']['PSNR'], 'SSIM':BUFFER['QHVS_FR_TOLIVEIRA']['SSIM'], 'BPP':BUFFER['QHVS_FR_TOLIVEIRA']['BPP']})
		results.append({'File name':file, 'Method':"QHVS_FC_TBRAHIMI", 'PSNR':BUFFER['QHVS_FC_TBRAHIMI']['PSNR'], 'SSIM':BUFFER['QHVS_FC_TBRAHIMI']['SSIM'], 'BPP':BUFFER['QHVS_FC_TBRAHIMI']['BPP']})
		results.append({'File name':file, 'Method':"QHVS_FF_TBRAHIMI", 'PSNR':BUFFER['QHVS_FF_TBRAHIMI']['PSNR'], 'SSIM':BUFFER['QHVS_FF_TBRAHIMI']['SSIM'], 'BPP':BUFFER['QHVS_FF_TBRAHIMI']['BPP']})
		results.append({'File name':file, 'Method':"QHVS_FR_TBRAHIMI", 'PSNR':BUFFER['QHVS_FR_TBRAHIMI']['PSNR'], 'SSIM':BUFFER['QHVS_FR_TBRAHIMI']['SSIM'], 'BPP':BUFFER['QHVS_FR_TBRAHIMI']['BPP']})
		results.append({'File name':file, 'Method':"QHVS_FC_TRAIZA", 'PSNR':BUFFER['QHVS_FC_TRAIZA']['PSNR'], 'SSIM':BUFFER['QHVS_FC_TRAIZA']['SSIM'], 'BPP':BUFFER['QHVS_FC_TRAIZA']['BPP']})
		results.append({'File name':file, 'Method':"QHVS_FF_TRAIZA", 'PSNR':BUFFER['QHVS_FF_TRAIZA']['PSNR'], 'SSIM':BUFFER['QHVS_FF_TRAIZA']['SSIM'], 'BPP':BUFFER['QHVS_FF_TRAIZA']['BPP']})
		results.append({'File name':file, 'Method':"QHVS_FR_TRAIZA", 'PSNR':BUFFER['QHVS_FR_TRAIZA']['PSNR'], 'SSIM':BUFFER['QHVS_FR_TRAIZA']['SSIM'], 'BPP':BUFFER['QHVS_FR_TRAIZA']['BPP']})

		
		
		
results = sorted(results, key=itemgetter('File name'))
destination = os.getcwd() + '/aplications/others/results/'

fieldnames = ['File name', 'Method', 'PSNR', 'SSIM', 'BPP']
with open(destination + 'teste_np2_F_forward_lc_transformations.csv', 'w') as csv_file:
	writer = csv.DictWriter(csv_file, fieldnames)
	writer.writeheader()
	for result in results:
		writer.writerow(result)
