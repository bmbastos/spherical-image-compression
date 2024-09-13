import os
import random
import csv
from numpy import *
from scipy import signal
from skimage.io import imread
import matplotlib.pyplot as plt
from pdb import set_trace as pause
from matrixes import *
from tools import *

def pre_processing(csv_file_name: str) -> tuple:
	data_set = []
	methods = []
	with open(csv_file_name, 'r') as file:
		reader = csv.DictReader(file)
		for row in reader:
			file_name, method, psnr, ssim, bpp = row.values()
			methods.append(method)
			data_set.append({
				'Filename': file_name,
				'Method': method,
				'PSNR': list(map(float, psnr.strip('[]').split(','))),
				'SSIM': list(map(float, ssim.strip('[]').split(','))),
				'BPP': list(map(float, bpp.strip('[]').split(',')))
			})
	methods = list(set(methods))  # Remover duplicatas de métodos
	return data_set, methods


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


def compute_scale_matrix(transformation_matrix:ndarray) -> matrix:
	scale_matrix = matrix(sqrt(linalg.inv(dot(transformation_matrix, transformation_matrix.T))))
	scale_vector = matrix(diag(scale_matrix))
	return scale_matrix, scale_vector
""" Matrix diagonal e elementos da matriz diagonal vetorizados """


def np2_round(quantization_matrix:matrix) -> matrix:
	return power(2, around(log2(quantization_matrix)))
""" Função que calcula as potencias de dois mais próximas de uma dada matriz - Oliveira """


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



# __MAIN__ #
target_file = "our_proposal.csv"
target_image = "Harbor"
target_bpp = 0.4

# Pre-processamento dos dados
dataset, methods = pre_processing(target_file)

qfs = {}

for data in dataset:
	if data['Filename'].startswith(target_image):
		nearest_bpp = min(data['BPP'], key=lambda num: abs(num - target_bpp))
		index_of_bpp = data['BPP'].index(nearest_bpp)
		qfs[data['Method']] = ((index_of_bpp + 1) * 5, nearest_bpp)
		#print(f'Método: {method}, Nearest BPP: {nearest_bpp}, Index: {index_of_bpp}')

# Exibir os QFs e BPPs
print(f"Métodos com os valores de QF e BPP mais próximos de {target_bpp}:")
print(qfs)
print()

path_images = "../ImagesForTest/Spherical"
T = calculate_matrix_of_transformation(8)
SO, so = compute_scale_matrix(TO)
SB, sb = compute_scale_matrix(TB)
SR, sr = compute_scale_matrix(TR)
ZO = dot(so.T, so)
ZB = dot(sb.T, sb)
ZR = dot(sr.T, sr)

files = os.listdir(path_images)
for file in files:
	if not file.startswith('Harbor'): continue
	full_path = os.path.join(path_images, file)
	if os.path.isfile(full_path) == False: continue

	image = imread(full_path, as_gray=True).astype(float)
	plt.axis('off'); plt.imshow(image, cmap='gray')
	#plt.savefig(file.split('.')[0] + '.png',format='png', bbox_inches='tight', pad_inches=0)
	plt.imsave(file.split('.')[0] + '.png', image, cmap='gray')
	plt.clf()
	pause()

	if image.max() <= 1:
		image = around(255*image)
	h, w = image.shape
	A = Tools.umount(image, (8, 8))# - 128

	ZT_tiled = tile(asarray([T]), (A.shape[0], 1, 1))
	ZO_tiled = tile(asarray([ZO]), (A.shape[0], 1, 1))
	ZB_tiled = tile(asarray([ZB]), (A.shape[0], 1, 1))
	ZR_tiled = tile(asarray([ZR]), (A.shape[0], 1, 1))


	print('DE SIMONE')
	QOliveira = adjust_quantization(qfs['DE_SIMONE'][0], Q0)
	QPhiOliveira = prepareQPhi(image, QOliveira)
	PhiPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T, A), T.T)
	PhiPrime2 = multiply(around(divide(PhiPrime1, QPhiOliveira)), QPhiOliveira)
	PhiPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, PhiPrime2), T)
	C = clip(Tools.remount(PhiPrime3, (h, w)), 0, 255)
	PhiPrime2 = PhiPrime2.reshape(h, w)
	print(WSPSNR(image, C))
	print(WSSSIM(image, C))
	print(qfs['DE_SIMONE'][1])
	plt.axis('off')
	plt.imshow(C, cmap='gray')
	#plt.savefig('IMG_DE_SIMONE_' + file.split('.')[0] + '.png', bbox_inches='tight', pad_inches=0)
	plt.imsave('IMG_DE_SIMONE_' + file.split('.')[0] + '.png', C, cmap='gray')
	plt.clf()
	del PhiPrime1; del PhiPrime2; del PhiPrime3; del C; del QPhiOliveira; del QOliveira; print()

	print('OLIVEIRA')
	QOliveira = adjust_quantization(qfs['OLIVEIRA'][0], Q0)
	OliveiraPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO, A), TO.T)
	QOliveiraForward = prepareQPhi(image, np2_round(divide(QOliveira, ZO)))
	QOliveiraBackward = prepareQPhi(image, np2_round(multiply(QOliveira, ZO)))
	OliveiraPrime2 = multiply(around(divide(OliveiraPrime1, QOliveiraForward)), QOliveiraBackward)
	OliveiraPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TO.T, OliveiraPrime2), TO)
	D = clip(Tools.remount(OliveiraPrime3, (h, w)), 0, 255)
	OliveiraPrime2 = OliveiraPrime2.reshape(h, w)
	print(WSPSNR(image, D))
	print(WSSSIM(image, D))
	print(qfs['OLIVEIRA'][1])
	plt.axis('off')
	plt.imshow(D, cmap='gray')
	#plt.savefig('IMG_OLIVEIRA_' + file.split('.')[0] + '.png', bbox_inches='tight', pad_inches=0)
	plt.imsave('IMG_OLIVEIRA_' + file.split('.')[0] + '.png', D, cmap='gray')
	plt.clf()
	del OliveiraPrime1; del OliveiraPrime2; del OliveiraPrime3; del D; del QOliveiraForward; del QOliveiraBackward; del QOliveira; print()

	print('RAIZA')
	QOliveira = adjust_quantization(qfs['RAIZA'][0], Q0)
	RaizaPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR, A), TR.T)
	QRaizaForward = prepareQPhi(image, np2_round(divide(QOliveira, ZR)))
	QRaizaBackward = prepareQPhi(image, np2_round(multiply(QOliveira, ZR)))
	RaizaPrime2 = multiply(around(divide(RaizaPrime1, QRaizaForward)), QRaizaBackward)
	RaizaPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TR.T, RaizaPrime2), TR)
	E = clip(Tools.remount(RaizaPrime3, (h, w)), 0, 255)
	RaizaPrime2 = RaizaPrime2.reshape(h, w)
	print(WSPSNR(image, E))
	print(WSSSIM(image, E))
	print(qfs['RAIZA'][1])
	plt.axis('off')
	plt.imshow(E, cmap='gray')
	#plt.savefig('IMG_RAIZA_' + file.split('.')[0] + '.png', bbox_inches='tight', pad_inches=0)
	plt.imsave('IMG_RAIZA_' + file.split('.')[0] + '.png', E, cmap='gray')
	plt.clf()
	del RaizaPrime1; del RaizaPrime2; del RaizaPrime3; del E; del QRaizaForward; del QRaizaBackward; del QOliveira; print()

	print('BRAHIMI')
	QOliveira = adjust_quantization(qfs['BRAHIMI'][0], Q0)
	BrahimiPrime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB, A), TB.T)
	QBrahimiForward = prepareQPhi(image, np2_round(divide(QOliveira, ZB)))
	QBrahimiBackward = prepareQPhi(image, np2_round(multiply(QOliveira, ZB)))
	BrahimiPrime2 = multiply(around(divide(BrahimiPrime1, QBrahimiForward)), QBrahimiBackward)
	BrahimiPrime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', TB.T, BrahimiPrime2), TB)
	F = clip(Tools.remount(BrahimiPrime3, (h, w)), 0, 255)
	BrahimiPrime2 = BrahimiPrime2.reshape(h, w)
	print(WSPSNR(image, F))
	print(WSSSIM(image, F))
	print(qfs['BRAHIMI'][1])
	plt.axis('off')
	plt.imshow(F, cmap='gray')
	#plt.savefig('IMG_BRAHIMI_' + file.split('.')[0] + '.png', bbox_inches='tight', pad_inches=0)
	plt.imsave('IMG_BRAHIMI_' + file.split('.')[0] + '.png', F, cmap='gray')
	plt.clf()
	del BrahimiPrime1; del BrahimiPrime2; del BrahimiPrime3; del F; del QBrahimiForward; del QBrahimiBackward; del QOliveira
