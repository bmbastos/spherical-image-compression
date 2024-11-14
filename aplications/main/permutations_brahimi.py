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
path_images = os.getcwd() + "/images_for_tests/spherical/by_resolution/4K"
T = calculate_matrix_of_transformation(8)
SO, so = compute_scale_matrix(TO)
SB, sb = compute_scale_matrix(TB)
SR, sr = compute_scale_matrix(TR)
ZO = dot(so.T, so)
ZB = dot(sb.T, sb)
ZR = dot(sr.T, sr)

permutations = [r'$\operatorname{np2}(\mathcal{P}(\mathbf{Q}, {\phi}) \Box \mathbf{Z})$',		# P2 = np2(adj(phi(Q), Z))
				r'$\mathcal{P}(\operatorname{np2}(\mathbf{Q} \Box \mathbf{Z}), {\phi})$',		# P3 = phi(np2(adj(Q, Z)))
				r'$\operatorname{np2}(\mathcal{P}((\mathbf{Q} \Box \mathbf{Z}), {\phi}))$']		# P4 = np2(phi(adj(Q, Z)))

quantization_factor = range(5, 100, 5)
results_4K = []
target = 1
processed_images = 0
files = os.listdir(path_images)
full_path = ''
for file in files:
	if not os.path.isdir(path_images + file) and file.endswith(".bmp"):
		#if processed_images == target:
			#break
		full_path = os.path.join(path_images, file)
		image = imread(full_path, as_gray=True).astype(float)
		#image = resize2(image, (128, 256), anti_aliasing=True)
		

		if image.max() <= 1:
			image = around(255*image)
		h, w = image.shape
		A = Tools.umount(image, (8, 8))# - 128
		print(f"{file} - {h}x{w} - {processed_images+1}/{len(files)}")

		ZT_tiled = tile(asarray([T]), (A.shape[0], 1, 1))
		ZO_tiled = tile(asarray([ZO]), (A.shape[0], 1, 1))
		ZB_tiled = tile(asarray([ZB]), (A.shape[0], 1, 1))
		ZR_tiled = tile(asarray([ZR]), (A.shape[0], 1, 1))

		BUFFER = {permutations[0]: {'PSNR':[], 'SSIM':[], 'BPP':[]},
					permutations[1]: {'PSNR':[], 'SSIM':[], 'BPP':[]},
					permutations[2]: {'PSNR':[], 'SSIM':[], 'BPP':[]}}

		for QF in quantization_factor:
			QOliveira = adjust_quantization(QF, Q0)
			Prime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB, A), TB.T)
			
			# P2 = np2(adj(phi(Q), Z))
			QP2Forward = np2_round(divide(prepareQPhi(image, QOliveira), ZB_tiled))
			QP2Backward = np2_round(multiply(prepareQPhi(image, QOliveira), ZB_tiled))
			P2Prime2 = multiply(around(divide(Prime1, QP2Forward)), QP2Backward)
			P2Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, P2Prime2), TB)
			C = clip(Tools.remount(P2Prime3, (h, w)), 0, 255)
			P2Prime2 = P2Prime2.reshape(h, w)
			BUFFER[permutations[0]]['PSNR'].append(WSPSNR(image, C))
			BUFFER[permutations[0]]['SSIM'].append(WSSSIM(image, C))
			BUFFER[permutations[0]]['BPP'].append(bpp(P2Prime2))
			del P2Prime2; del P2Prime3; del QP2Forward; del QP2Backward; del C

			# P3 = phi(np2(adj(Q, Z)))
			QP3Forward = prepareQPhi(image, np2_round(divide(QOliveira, ZB)))
			QP3Backward = prepareQPhi(image, np2_round(multiply(QOliveira, ZB)))
			P3Prime2 = multiply(around(divide(Prime1, QP3Forward)), QP3Backward)
			P3Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, P3Prime2), TB)
			D = clip(Tools.remount(P3Prime3, (h, w)), 0, 255)
			P3Prime2 = P3Prime2.reshape(h, w)
			BUFFER[permutations[1]]['PSNR'].append(WSPSNR(image, D))
			BUFFER[permutations[1]]['SSIM'].append(WSSSIM(image, D))
			BUFFER[permutations[1]]['BPP'].append(bpp(P3Prime2))
			del P3Prime2; del P3Prime3; del QP3Forward; del QP3Backward; del D

			# P4 = np2(phi(adj(Q, Z)))
			QP4Forward = np2_round(prepareQPhi(image, divide(QOliveira, ZB)))
			QP4Backward = np2_round(prepareQPhi(image, multiply(QOliveira, ZB)))
			P4Prime2 = multiply(around(divide(Prime1, QP4Forward)), QP4Backward)
			P4Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, P4Prime2), TB)
			E = clip(Tools.remount(P4Prime3, (h, w)), 0, 255)
			P4Prime2 = P4Prime2.reshape(h, w)
			BUFFER[permutations[2]]['PSNR'].append(WSPSNR(image, E))
			BUFFER[permutations[2]]['SSIM'].append(WSSSIM(image, E))
			BUFFER[permutations[2]]['BPP'].append(bpp(P4Prime2))
			del P4Prime2; del P4Prime3; del QP4Forward; del QP4Backward; del E

		processed_images += 1
		results_4K.append({'File name':file, 'Method':permutations[0], 'PSNR':BUFFER[permutations[0]]['PSNR'], 'SSIM':BUFFER[permutations[0]]['SSIM'], 'BPP':BUFFER[permutations[0]]['BPP']})
		results_4K.append({'File name':file, 'Method':permutations[1], 'PSNR':BUFFER[permutations[1]]['PSNR'], 'SSIM':BUFFER[permutations[1]]['SSIM'], 'BPP':BUFFER[permutations[1]]['BPP']})
		results_4K.append({'File name':file, 'Method':permutations[2], 'PSNR':BUFFER[permutations[2]]['PSNR'], 'SSIM':BUFFER[permutations[2]]['SSIM'], 'BPP':BUFFER[permutations[2]]['BPP']})
		
results_4K = sorted(results_4K, key=itemgetter('File name'))
destination = os.getcwd() + '/aplications/main/results/'
fieldnames = ['File name', 'Method', 'PSNR', 'SSIM', 'BPP']
with open(destination + 'permutations_brahimi_4K.csv', 'w') as csv_file_4k:
	writer_4k = csv.DictWriter(csv_file_4k, fieldnames)
	writer_4k.writeheader()
	for result in results_4K:
		writer_4k.writerow(result)

print('Processamento finalizado')


