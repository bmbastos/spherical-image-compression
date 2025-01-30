from numpy import *
from skimage.io import imread
from pdb import set_trace as pause
from ..mylibs.matrixes import *
from ..mylibs.tools import Tools
import os

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


def prepareQPhi(image:ndarray, quantization_matrix:ndarray, adjustment_coefficients:ndarray, N = 8):
	h, w = image.shape
	k_lut, min_lut, max_lut = build_LUT(h)
	print(f"Altura: {h}, Largura: {w}")
	elevation_alvo = pi/8
	if elevation_alvo < -pi/2 or elevation_alvo > pi/2:
		print('Elevação alvo fora do intervalo [-pi/2, pi/2]')
		return
	q_alvo_f = np2_round(divide(QtildeAtEl(k_lut, min_lut, max_lut, elevation_alvo, quantization_matrix), adjustment_coefficients))
	q_alvo_b = np2_round(multiply(QtildeAtEl(k_lut, min_lut, max_lut, elevation_alvo, quantization_matrix), adjustment_coefficients))
	print('Quantization matrix:')
	print(quantization_matrix)
	print('Scaling matrix:')
	print(adjustment_coefficients)
	print('Q_forward:')
	print(q_alvo_f)
	print()
	print('Q_backward:')
	print(q_alvo_b)
	els = linspace(-pi/2, pi/2, h//N+1)
	els = 0.5*(els[1:] + els[:-1]) # gets the "central" block elevation
	QPhi = []
	for el in els: 
		QPhi.append(QtildeAtEl(k_lut, min_lut, max_lut, el, quantization_matrix))
	QPhi = asarray(QPhi)
	QPhi = repeat(QPhi, w//N, axis=0)
	#plot.imshow(tools.Tools.remount(QPhi, (h, w))); plot.show() # plot the quantization matrices map
	return QPhi


##__MAIN__##
path_images = os.getcwd() + "/images_for_tests/spherical/by_resolution/4K/"
file_init = "AerialCity"
files = os.listdir(path_images)
J_8 = matrix(ones((8, 8), dtype=float))
for file in files:
	if not file.startswith(file_init): continue
	full_path = path_images + file
	image = imread(full_path, as_gray=True).astype(float)
	if image.max() <= 1:
				image = around(255*image)
	print(f"Nome de arquivo: {file}")
	h, w = image.shape
	quality_factor = 75
	SR, sr = compute_scale_matrix(TR)
	ZR = dot(sr.T, sr)
	Q = adjust_quantization(quality_factor, Q0)
	QPhi = prepareQPhi(image, Q, ZR)


