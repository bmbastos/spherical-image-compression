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
		#image = resize2(image, (512, 1024), anti_aliasing=True)

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

		BUFFER = {'QJPEG_CC': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QJPEG_CF': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QJPEG_CR': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QBRAHIMI_CC': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QBRAHIMI_CF': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QBRAHIMI_CR': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QHVS_CC': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QHVS_CF': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'QHVS_CR': {'PSNR':[], 'SSIM':[], 'BPP':[]}}


		for QF in tqdm(quantization_factor):
			''' QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG -- QJPEG'''
			QOliveira = adjust_quantization(QF, Q0)
			QPhi_Oliveira_Forward = prepareQPhi(image, np2_ceil(QOliveira))
			# CEIL FORWARD + CEIL BACKWARD (CC)
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_ceil(QOliveira))
			J_CC_Prime2 = multiply(around(divide(JpegPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_CC_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, J_CC_Prime2), T)
			J_CC = clip(Tools.remount(J_CC_Prime3, (h, w)), 0, 255)
			J_CC_Prime2 = J_CC_Prime2.reshape(h, w)
			BUFFER['QJPEG_CC']['PSNR'].append(WSPSNR(image, J_CC))
			BUFFER['QJPEG_CC']['SSIM'].append(WSSSIM(image, J_CC))
			BUFFER['QJPEG_CC']['BPP'].append(bpp(J_CC_Prime2))
			del J_CC_Prime2, J_CC_Prime3, J_CC, QPhi_Oliveira_Backward

			# CEIL FORWARD + FLOOR BACKWARD (CF)
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_floor(QOliveira))
			J_CF_Prime2 = multiply(around(divide(JpegPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_CF_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, J_CF_Prime2), T)
			J_CF = clip(Tools.remount(J_CF_Prime3, (h, w)), 0, 255)
			J_CF_Prime2 = J_CF_Prime2.reshape(h, w)
			BUFFER['QJPEG_CF']['PSNR'].append(WSPSNR(image, J_CF))
			BUFFER['QJPEG_CF']['SSIM'].append(WSSSIM(image, J_CF))
			BUFFER['QJPEG_CF']['BPP'].append(bpp(J_CF_Prime2))
			del J_CF_Prime2, J_CF_Prime3, J_CF, QPhi_Oliveira_Backward

			# CEIL FORWARD + ROUND BACKWARD (CR)
			QPhi_Oliveira_Backward = prepareQPhi(image, np2_round(QOliveira))
			J_CR_Prime2 = multiply(around(divide(JpegPrime1, QPhi_Oliveira_Forward)), QPhi_Oliveira_Backward)
			J_CR_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, J_CR_Prime2), T)
			J_CR = clip(Tools.remount(J_CR_Prime3, (h, w)), 0, 255)
			J_CR_Prime2 = J_CR_Prime2.reshape(h, w)
			BUFFER['QJPEG_CR']['PSNR'].append(WSPSNR(image, J_CR))
			BUFFER['QJPEG_CR']['SSIM'].append(WSSSIM(image, J_CR))
			BUFFER['QJPEG_CR']['BPP'].append(bpp(J_CR_Prime2))
			del J_CR_Prime2, J_CR_Prime3, J_CR, QPhi_Oliveira_Backward, QPhi_Oliveira_Forward
			
			''' QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI -- QBRAHIMI'''
			QBrahimi = adjust_quantization(QF, QB)
			QPhi_Brahimi_Forward = prepareQPhi(image, np2_ceil(QBrahimi))
			# CEIL FORWARD + CEIL BACKWARD (CC)
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_ceil(QBrahimi))
			B_CC_Prime2 = multiply(around(divide(JpegPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_CC_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, B_CC_Prime2), T)
			B_CC = clip(Tools.remount(B_CC_Prime3, (h, w)), 0, 255)
			B_CC_Prime2 = B_CC_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_CC']['PSNR'].append(WSPSNR(image, B_CC))
			BUFFER['QBRAHIMI_CC']['SSIM'].append(WSSSIM(image, B_CC))
			BUFFER['QBRAHIMI_CC']['BPP'].append(bpp(B_CC_Prime2))
			del B_CC_Prime2, B_CC_Prime3, B_CC, QPhi_Brahimi_Backward

			# CEIL FORWARD + FLOOR BACKWARD (CF)
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_floor(QBrahimi))
			B_CF_Prime2 = multiply(around(divide(JpegPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_CF_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, B_CF_Prime2), T)
			B_CF = clip(Tools.remount(B_CF_Prime3, (h, w)), 0, 255)
			B_CF_Prime2 = B_CF_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_CF']['PSNR'].append(WSPSNR(image, B_CF))
			BUFFER['QBRAHIMI_CF']['SSIM'].append(WSSSIM(image, B_CF))
			BUFFER['QBRAHIMI_CF']['BPP'].append(bpp(B_CF_Prime2))
			del B_CF_Prime2, B_CF_Prime3, B_CF, QPhi_Brahimi_Backward

			# CEIL FORWARD + ROUND BACKWARD (CR)
			QPhi_Brahimi_Backward = prepareQPhi(image, np2_round(QBrahimi))
			B_CR_Prime2 = multiply(around(divide(JpegPrime1, QPhi_Brahimi_Forward)), QPhi_Brahimi_Backward)
			B_CR_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, B_CR_Prime2), T)
			B_CR = clip(Tools.remount(B_CR_Prime3, (h, w)), 0, 255)
			B_CR_Prime2 = B_CR_Prime2.reshape(h, w)
			BUFFER['QBRAHIMI_CR']['PSNR'].append(WSPSNR(image, B_CR))
			BUFFER['QBRAHIMI_CR']['SSIM'].append(WSSSIM(image, B_CR))
			BUFFER['QBRAHIMI_CR']['BPP'].append(bpp(B_CR_Prime2))
			del B_CR_Prime2, B_CR_Prime3, B_CR, QPhi_Brahimi_Backward, QPhi_Brahimi_Forward
			
			''' QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS -- QHVS'''
			QHvs = adjust_quantization(QF, Qhvs)
			QPhi_Hvs_Forward = prepareQPhi(image, np2_ceil(QHvs))
			# CEIL FORWARD + CEIL BACKWARD (CC)
			QPhi_Hvs_Backward = prepareQPhi(image, np2_ceil(QHvs))
			H_CC_Prime2 = multiply(around(divide(JpegPrime1, QPhi_Hvs_Forward)), QPhi_Hvs_Backward)
			H_CC_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, H_CC_Prime2), T)
			H_CC = clip(Tools.remount(H_CC_Prime3, (h, w)), 0, 255)
			H_CC_Prime2 = H_CC_Prime2.reshape(h, w)
			BUFFER['QHVS_CC']['PSNR'].append(WSPSNR(image, H_CC))
			BUFFER['QHVS_CC']['SSIM'].append(WSSSIM(image, H_CC))
			BUFFER['QHVS_CC']['BPP'].append(bpp(H_CC_Prime2))
			del H_CC_Prime2, H_CC_Prime3, H_CC, QPhi_Hvs_Backward

			# CEIL FORWARD + FLOOR BACKWARD (CF)
			QPhi_Hvs_Backward = prepareQPhi(image, np2_floor(QHvs))
			H_CF_Prime2 = multiply(around(divide(JpegPrime1, QPhi_Hvs_Forward)), QPhi_Hvs_Backward)
			H_CF_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, H_CF_Prime2), T)
			H_CF = clip(Tools.remount(H_CF_Prime3, (h, w)), 0, 255)
			H_CF_Prime2 = H_CF_Prime2.reshape(h, w)
			BUFFER['QHVS_CF']['PSNR'].append(WSPSNR(image, H_CF))
			BUFFER['QHVS_CF']['SSIM'].append(WSSSIM(image, H_CF))
			BUFFER['QHVS_CF']['BPP'].append(bpp(H_CF_Prime2))
			del H_CF_Prime2, H_CF_Prime3, H_CF, QPhi_Hvs_Backward

			# CEIL FORWARD + ROUND BACKWARD (CR)
			QPhi_Hvs_Backward = prepareQPhi(image, np2_round(QHvs))
			H_CR_Prime2 = multiply(around(divide(JpegPrime1, QPhi_Hvs_Forward)), QPhi_Hvs_Backward)
			H_CR_Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, H_CR_Prime2), T)
			H_CR = clip(Tools.remount(H_CR_Prime3, (h, w)), 0, 255)
			H_CR_Prime2 = H_CR_Prime2.reshape(h, w)
			BUFFER['QHVS_CR']['PSNR'].append(WSPSNR(image, H_CR))
			BUFFER['QHVS_CR']['SSIM'].append(WSSSIM(image, H_CR))
			BUFFER['QHVS_CR']['BPP'].append(bpp(H_CR_Prime2))
			del H_CR_Prime2, H_CR_Prime3, H_CR, QPhi_Hvs_Backward, QPhi_Hvs_Forward

			
		processed_images += 1
		results.append({'File name':file, 'Method':"JPEG_CC", 'PSNR':BUFFER['QJPEG_CC']['PSNR'], 'SSIM':BUFFER['QJPEG_CC']['SSIM'], 'BPP':BUFFER['QJPEG_CC']['BPP']})
		results.append({'File name':file, 'Method':"JPEG_CF", 'PSNR':BUFFER['QJPEG_CF']['PSNR'], 'SSIM':BUFFER['QJPEG_CF']['SSIM'], 'BPP':BUFFER['QJPEG_CF']['BPP']})
		results.append({'File name':file, 'Method':"JPEG_CR", 'PSNR':BUFFER['QJPEG_CR']['PSNR'], 'SSIM':BUFFER['QJPEG_CR']['SSIM'], 'BPP':BUFFER['QJPEG_CR']['BPP']})
		results.append({'File name':file, 'Method':"Brahimi_CC", 'PSNR':BUFFER['QBRAHIMI_CC']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_CC']['SSIM'], 'BPP':BUFFER['QBRAHIMI_CC']['BPP']})
		results.append({'File name':file, 'Method':"Brahimi_CF", 'PSNR':BUFFER['QBRAHIMI_CF']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_CF']['SSIM'], 'BPP':BUFFER['QBRAHIMI_CF']['BPP']})
		results.append({'File name':file, 'Method':"Brahimi_CR", 'PSNR':BUFFER['QBRAHIMI_CR']['PSNR'], 'SSIM':BUFFER['QBRAHIMI_CR']['SSIM'], 'BPP':BUFFER['QBRAHIMI_CR']['BPP']})
		results.append({'File name':file, 'Method':"HVS_CC", 'PSNR':BUFFER['QHVS_CC']['PSNR'], 'SSIM':BUFFER['QHVS_CC']['SSIM'], 'BPP':BUFFER['QHVS_CC']['BPP']})
		results.append({'File name':file, 'Method':"HVS_CF", 'PSNR':BUFFER['QHVS_CF']['PSNR'], 'SSIM':BUFFER['QHVS_CF']['SSIM'], 'BPP':BUFFER['QHVS_CF']['BPP']})
		results.append({'File name':file, 'Method':"HVS_CR", 'PSNR':BUFFER['QHVS_CR']['PSNR'], 'SSIM':BUFFER['QHVS_CR']['SSIM'], 'BPP':BUFFER['QHVS_CR']['BPP']})
		
		
		
results = sorted(results, key=itemgetter('File name'))
destination = os.getcwd() + '/aplications/others/results/'

fieldnames = ['File name', 'Method', 'PSNR', 'SSIM', 'BPP']
with open(destination + 'teste_np2_C_forward.csv', 'w') as csv_file:
	writer = csv.DictWriter(csv_file, fieldnames)
	writer.writeheader()
	for result in results:
		writer.writerow(result)
