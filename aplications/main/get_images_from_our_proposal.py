import os
import random
import csv
from numpy import *
from builtins import min as mymin
from scipy import signal
from skimage.io import imread
import matplotlib.pyplot as plt
from pdb import set_trace as pause
from ..mylibs.matrixes import *
from ..mylibs.tools import Tools

def pre_processing(csv_file_name: str) -> tuple:
	data_set = []
	methods = []
	with open(csv_file_name, 'r') as file:
		reader = csv.DictReader(file)
		for row in reader:
			file_name, method, psnr, ssim, bpp = row.values()
			methods.append(method)

			# Helper function to clean and convert values
			def clean_and_convert(value_str):
				return list(map(float, value_str.replace('np.float64(', '').replace(')', '').strip('[]').split(',')))

			data_set.append({
				'Filename': file_name,
				'Method': method,
				'PSNR': clean_and_convert(psnr),
				'SSIM': clean_and_convert(ssim),
				'BPP': clean_and_convert(bpp)
			})
	
	methods = list(set(methods))
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

def np2_ceil(quantization_matrix:matrix) -> matrix:
	return power(2, ceil(log2(quantization_matrix))) # Don't use when the transform has low-complexity
""" Função de transformação de uma matriz em uma matriz de potências de dois - Brahimi """

def np2_floor(quantization_matrix:matrix) -> matrix:
	return power(2, floor(log2(quantization_matrix))) # Don't use when the transform has low-complexity
""" Função de transformação de uma matriz em uma matriz de potências de dois - Our """

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
print('O que será alterado?\n1) Nada\n2) Transformada\n3) Quantização\n4) Função NP2\n')
sel1 = int(input('Digite a opção desejada: '))
sel2 = 0
target_file = "our_new_proposal_4K.csv"
if sel1 == 1:
	target_file = "aplications/main/results/our_new_proposal_4K.csv"
elif sel1 == 2:
	print('Qual transformada vai usar?\n1) Oliveira\n2) Brahimi\n')
	sel2 = int(input('Digite a opção desejada: '))
	if sel2 == 1:
		target_file = "aplications/others/results/permutation_transformation_Oliveira_4K.csv"
	elif sel2 == 2:
		target_file = "aplications/others/results/permutation_transformation_Brahimi_4K.csv"
	else:
		print('Opção inválida')
		exit()
elif sel1 == 3:
	print('Qual quantização vai usar?\n1) Brahimi\n2) HVS\n')
	sel2 = int(input('Digite para qual quantização será gerada as imagens (1 - Brahimi, 2 - HVS): '))
	if sel2 == 1:
		target_file = "aplications/others/results/permutation_quantization_Brahimi_4K.csv"
	elif sel2 == 2:
		target_file = "aplications/others/results/permutation_quantization_HVS_4K.csv"
	else:
		print('Opção inválida')
		exit()
elif sel1 == 4:
	print('Qual função NP2 vai usar?\n1) ceil\n2) floor\n')
	sel2 = int(input('Digite a opção desejada: '))
	if sel2 == 1:
		target_file = "aplications/others/results/permutation_np2_function_ceil_4K.csv"
	elif sel2 == 2:
		target_file = "aplications/others/results/permutation_np2_function_floor_4K.csv"
else:
	print('Opção inválida')
	exit()


target_image = "AerialCity_3840x1920_30fps_8bit_420_erp_0"
target_bpp = 0.4

# Pre-processamento dos dados
dataset, methods = pre_processing(target_file)

qfs = {}

for data in dataset:
	if data['Filename'].startswith(target_image):
		print(data['Method'])
		print(data['BPP'])
		nearest_bpp = mymin(data['BPP'], key=lambda num: abs(num - target_bpp))
		index_of_bpp = data['BPP'].index(nearest_bpp)
		qfs[data['Method']] = ((index_of_bpp + 1) * 5, nearest_bpp)
		#print(f'Método: {method}, Nearest BPP: {nearest_bpp}, Index: {index_of_bpp}')

# Exibir os QFs e BPPs
print(f"Métodos com os valores de QF e BPP mais próximos de {target_bpp}:")
print(qfs)
print()

path_images = os.getcwd() + "/images_for_tests/spherical/by_resolution/4K"
T = calculate_matrix_of_transformation(8)
SO, so = compute_scale_matrix(TO)
SB, sb = compute_scale_matrix(TB)
SR, sr = compute_scale_matrix(TR)
ZO = dot(so.T, so)
ZB = dot(sb.T, sb)
ZR = dot(sr.T, sr)


T1 = TR
Z1 = ZR
Q1 = Q0
transformation = 'TR'
quantization = 'Q0'
np2_function = 'round'
if sel1 == 2:
	if sel2 == 0:
		print(f'sel1 = {sel1}: Erro em sel2 {sel2}')
		exit()
	elif sel2 == 1:
		transformation = 'TO'
		T1 = TO
		Z1 = ZO
	elif sel2 == 2:
		transformation = 'TB'
		T1 = TB
		Z1 = ZB
	else:
		print('Erro em sel2')
		exit()
elif sel1 == 3:
	if sel2 == 0:
		print(f'sel1 = {sel1}: Erro em sel2 {sel2}')
		exit()
	elif sel2 == 1:
		quantization = 'QB'
		Q1 = QB
	elif sel2 == 2:
		quantization = 'QHVS'
		Q1 = Qhvs
	else:
		print('Erro em sel2')
		exit()
elif sel1 == 4:
	if sel2 == 0:
		print(f'sel1 = {sel1}: Erro em sel2 {sel2}')
		exit()
	elif sel2 == 1:
		np2_function = 'ceil'
	elif sel2 == 2:
		np2_function = 'floor'
elif sel1 != 1:
	print('Erro em sel1')
	exit()

pause()

print(f'Analisando {target_file} com transformação {transformation}, quantização {quantization} e função NP2_{np2_function}')

files = os.listdir(path_images)
for file in files:
	if not file.startswith(target_image): continue
	full_path = os.path.join(path_images, file)
	if os.path.isfile(full_path) == False: continue

	image = imread(full_path, as_gray=True).astype(float)
	plt.axis('off'); plt.imshow(image, cmap='gray')
	plt.imsave('IMG_ORIGINAL_' + file.split('.')[0] + '.png', image, cmap='gray')
	plt.clf()

	if image.max() <= 1:
		image = around(255*image)
	h, w = image.shape
	A = Tools.umount(image, (8, 8))# - 128

	ZT_tiled = tile(asarray([T]), (A.shape[0], 1, 1))
	ZO_tiled = tile(asarray([ZO]), (A.shape[0], 1, 1))
	ZB_tiled = tile(asarray([ZB]), (A.shape[0], 1, 1))
	ZR_tiled = tile(asarray([ZR]), (A.shape[0], 1, 1))


	'''
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
	plt.imsave('IMG_DE_SIMONE_' + file.split('.')[0] + '.png', C, cmap='gray')
	plt.clf()
	del PhiPrime1; del PhiPrime2; del PhiPrime3; del C; del QPhiOliveira; del QOliveira; print()

	print('P2')
	QAdjustedP2 = adjust_quantization(qfs['OUR_P2'][0], Q1)
	P2Prime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T1, A), T1.T)
	QPhiP2Forward = 0
	QPhiP2Backward = 0
	if sel1 == 4 and sel2 == 1:
		QPhiP2Forward = np2_ceil(divide(prepareQPhi(image, QAdjustedP2), ZR_tiled))
		QPhiP2Backward = np2_ceil(multiply(prepareQPhi(image, QAdjustedP2), ZR_tiled))
	elif sel1 == 4 and sel2 == 2:
		QPhiP2Forward = np2_floor(divide(prepareQPhi(image, QAdjustedP2), ZR_tiled))
		QPhiP2Backward = np2_floor(multiply(prepareQPhi(image, QAdjustedP2), ZR_tiled))
	else:
		QPhiP2Forward = np2_round(divide(prepareQPhi(image, QAdjustedP2), ZR_tiled))
		QPhiP2Backward = np2_round(multiply(prepareQPhi(image, QAdjustedP2), ZR_tiled))
	P2Prime2 = multiply(around(divide(P2Prime1, QPhiP2Forward)), QPhiP2Backward)
	P2Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T1.T, P2Prime2), T1)
	P2 = clip(Tools.remount(P2Prime3, (h, w)), 0, 255)
	P2Prime2 = P2Prime2.reshape(h, w)
	print(WSPSNR(image, P2))
	print(WSSSIM(image, P2))
	print(qfs['OUR_P2'][1])
	plt.axis('off')
	plt.imshow(P2, cmap='gray')
	plt.imsave('IMG_OUR_P2_' + transformation + '_' + quantization + '_' + file.split('.')[0] + '.png', P2, cmap='gray')
	plt.clf()
	del P2Prime1; del P2Prime2; del P2Prime3; del P2; del QPhiP2Forward; del QPhiP2Backward; del QAdjustedP2; print()
	'''
	
	print('P3')
	QAdjustedP3 = adjust_quantization(qfs['OUR_P3'][0], Q1)
	P3Prime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T1, A), T1.T)
	QPhiP3Forward = 0
	QPhiP3Backward = 0
	if sel1 == 4 and sel2 == 1:
		QPhiP3Forward = prepareQPhi(image, np2_ceil(divide(QAdjustedP3, Z1)))
		QPhiP3Backward = prepareQPhi(image, np2_ceil(multiply(QAdjustedP3, Z1)))
	elif sel1 == 4 and sel2 == 2:
		QPhiP3Forward = prepareQPhi(image, np2_floor(divide(QAdjustedP3, Z1)))
		QPhiP3Backward = prepareQPhi(image, np2_floor(multiply(QAdjustedP3, Z1)))
	else:
		QPhiP3Forward = prepareQPhi(image, np2_round(divide(QAdjustedP3, Z1)))
		QPhiP3Backward = prepareQPhi(image, np2_round(multiply(QAdjustedP3, Z1)))
	P3Prime2 = multiply(around(divide(P3Prime1, QPhiP3Forward)), QPhiP3Backward)
	P3Prime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T1.T, P3Prime2), T1)
	P3 = clip(Tools.remount(P3Prime3, (h, w)), 0, 255)
	P3Prime2 = P3Prime2.reshape(h, w)
	print(WSPSNR(image, P3))
	print(WSSSIM(image, P3))
	print(qfs['OUR_P3'][1])
	plt.axis('off')
	plt.imshow(P3, cmap='gray')
	plt.imsave('IMG_OUR_P3_' + transformation + '_' + quantization + '_' + np2_function + '_' + file.split('.')[0] + '.png', P3, cmap='gray')
	plt.clf()
	del P3Prime1; del P3Prime2; del P3Prime3; del P3; del QPhiP3Forward; del QPhiP3Backward; del QAdjustedP3; print()

	print('Imagens geradas para nossa nova proposta')