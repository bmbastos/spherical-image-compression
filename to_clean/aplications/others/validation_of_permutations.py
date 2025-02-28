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
path_images = os.getcwd() + "/images_for_tests/spherical/by_resolution/4K/"
T = calculate_matrix_of_transformation(8)
SO, so = compute_scale_matrix(TO)
#SB, sb = compute_scale_matrix(TB)
#SR, sr = compute_scale_matrix(TR)
ZO = dot(so.T, so)
#ZB = dot(sb.T, sb)
#ZR = dot(sr.T, sr)
#pause()

quantization_factor = range(5, 100, 5)
results = []
target = 1
processed_images = 0
files = os.listdir(path_images)
file_target = "Aerial"
full_path = ''
for file in files:
	if not file.startswith(file_target): continue
	if not os.path.isdir(full_path) and file.endswith(".bmp"):
		if processed_images == target:
			break
		full_path = os.path.join(path_images, file)
		image = imread(full_path, as_gray=True).astype(float)
		image = resize2(image, (512, 1024), anti_aliasing=True)
		

		if image.max() <= 1:
			image = around(255*image)
		h, w = image.shape
		A = Tools.umount(image, (8, 8))# - 128
		print(f"{file} - {h}x{w} - {processed_images+1}/{len(files)}")

		ZT_tiled = tile(asarray([T]), (A.shape[0], 1, 1))
		ZO_tiled = tile(asarray([ZO]), (A.shape[0], 1, 1))
		#ZB_tiled = tile(asarray([ZB]), (A.shape[0], 1, 1))
		#ZR_tiled = tile(asarray([ZR]), (A.shape[0], 1, 1))

		BUFFER = {'A': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'B': {'PSNR':[], 'SSIM':[], 'BPP':[]},
					'C': {'PSNR':[], 'SSIM':[], 'BPP':[]}}


		for QF in quantization_factor:
			print(QF)
			QOliveira = adjust_quantization(QF, Q0)

			# Permutation 1 - De Simone + RDCT
			P1Prime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', dot(SO, TO), A), dot(SO, TO).T)
			QPhiP1Forward = prepareQPhi(image, QOliveira)
			QPhiP1Backward = prepareQPhi(image, QOliveira)
			P1Prime2 = multiply(around(divide(P1Prime1, QPhiP1Forward)), QPhiP1Backward)
			P1Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', dot(SO, TO).T, P1Prime2), dot(SO, TO))
			P1 = clip(Tools.remount(P1Prime3, (h, w)), 0, 255)
			P1Prime2 = P1Prime2.reshape(h, w)
			BUFFER['C']['PSNR'].append(WSPSNR(image, P1))
			BUFFER['C']['SSIM'].append(WSSSIM(image, P1))
			BUFFER['C']['BPP'].append(bpp(P1Prime2))
			del P1Prime1; del P1Prime2; del P1Prime3; del P1; del QPhiP1Forward; del QPhiP1Backward

			
			# Permutation 2 - De Simone + RDCT
			P2Prime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO, A), TO.T)
			QPhiP2Forward = divide(prepareQPhi(image, QOliveira), ZO_tiled)
			QPhiP2Backward = multiply(prepareQPhi(image, QOliveira), ZO_tiled)
			P2Prime2 = multiply(around(divide(P2Prime1, QPhiP2Forward)), QPhiP2Backward)
			P2Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, P2Prime2), TO)
			P2 = clip(Tools.remount(P2Prime3, (h, w)), 0, 255)
			P2Prime2 = P2Prime2.reshape(h, w)
			BUFFER['A']['PSNR'].append(WSPSNR(image, P2))
			BUFFER['A']['SSIM'].append(WSSSIM(image, P2))
			BUFFER['A']['BPP'].append(bpp(P2Prime2))
			del P2Prime1; del P2Prime2; del P2Prime3; del P2; del QPhiP2Forward; del QPhiP2Backward

			# Permutation 3 
			P3Prime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO, A), TO.T)
			QPhiP3Forward = prepareQPhi(image, divide(QOliveira, ZO))
			QPhiP3Backward = prepareQPhi(image, multiply(QOliveira, ZO))
			P3Prime2 = multiply(around(divide(P3Prime1, QPhiP3Forward)), QPhiP3Backward)
			P3Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, P3Prime2), TO)
			P3 = clip(Tools.remount(P3Prime3, (h, w)), 0, 255)
			P3Prime2 = P3Prime2.reshape(h, w)
			BUFFER['B']['PSNR'].append(WSPSNR(image, P3))
			BUFFER['B']['SSIM'].append(WSSSIM(image, P3))
			BUFFER['B']['BPP'].append(bpp(P3Prime2))
			del P3Prime1; del P3Prime2; del P3Prime3; del P3; del QPhiP3Forward; del QPhiP3Backward



		processed_images += 1
		results.append({'File name':file, 'Method':"A", 'PSNR':BUFFER['A']['PSNR'], 'SSIM':BUFFER['A']['SSIM'], 'BPP':BUFFER['A']['BPP']})
		results.append({'File name':file, 'Method':"B", 'PSNR':BUFFER['B']['PSNR'], 'SSIM':BUFFER['B']['SSIM'], 'BPP':BUFFER['B']['BPP']})
		results.append({'File name':file, 'Method':"C", 'PSNR':BUFFER['C']['PSNR'], 'SSIM':BUFFER['C']['SSIM'], 'BPP':BUFFER['C']['BPP']})
processed_images = 0

results = sorted(results, key=itemgetter('File name'))

destination = os.getcwd() + '/aplications/others/results/'
fieldnames = ['File name', 'Method', 'PSNR', 'SSIM', 'BPP']
with open(destination + 'validation_permutations.csv', 'w') as csv_file_4k:
	writer_4k = csv.DictWriter(csv_file_4k, fieldnames)
	writer_4k.writeheader()
	for result in results:
		writer_4k.writerow(result)

print('Processamento finalizado')
''' Teste de validade se P2 é matematicamente correta '''

