import os
from numpy import *
from time import time
from skimage.io import imread
from matplotlib import pyplot as plot
from pdb import set_trace as pause

"""
Principal implementação onde será embedado as ideias de compressão de imagens
visando a minimização do custo para imagens omnidirecionais
"""

# DEFINIÇÃO DE FUNÇÕES

# Realiza o cálculo de uma matriz de tranformação para JPEG com base no tamanho 'k' do bloco
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

# Aplicação do fator de quantização com base no QF (Quality-factor) e na matriz usada
def quantization(quality_factor:int, quantization_matrix:ndarray) -> ndarray:
	s = 0.0
	if quality_factor < 50:
		s = 5_000 / quality_factor
	else:
		s = 200 - (2 * quality_factor)
	resulting_matrix = floor((s * quantization_matrix + 50) / 100)
	return resulting_matrix

# Aplicação da transformada exata 
def apply_exact_transform(transformation_matrix:ndarray, image:ndarray, k:int) -> ndarray:
	nrows, ncols = image.shape
	dct_image = zeros((nrows, ncols), dtype=float32)
	row = 0
	while(row < nrows):
		col = 0
		while(col < ncols):
			dct_image[row:row+k, col:col+k] = dot(dot(transformation_matrix, image[row:row+k, col:col+k]), transformation_matrix.T)
			col += k
		row += k
	return dct_image

# Aplicação da transformada inversa
def apply_inverse_transform(transformation_matrix:ndarray, quantized_image:ndarray, k:int) -> ndarray:
	nrows, ncols = quantized_image.shape
	idct_image = zeros((nrows, ncols), dtype=float32)
	row = 0
	while(row < nrows):
		col = 0
		while(col < ncols):
			idct_image[row:row+k, col:col+k] = dot(dot(transformation_matrix.T, quantized_image[row:row+k, col:col+k]), transformation_matrix)
			col += k
		row += k
	return clip(idct_image, 0, 255)

# Aplica a quantização padrão JPEG
def apply_quantization(quantization_matrix:ndarray, dct_image:ndarray, k:int) -> ndarray:
	nrows, ncols = dct_image.shape
	quantized_image = zeros((nrows, ncols), dtype=float32)
	row = 0
	while(row < nrows):
		col = 0
		while(col < ncols):
			quantized_image[row:row+k, col:col+k] = multiply(around(divide(dct_image[row:row+k, col:col+k], quantization_matrix)), quantization_matrix)
			col += k
		row += k
	return quantized_image

# Aplica a quantização sugerida por oliveira
def apply_oliveira_quantization(q_f_matrix:matrix, q_i_matrix:matrix, dct_image:ndarray, k:int) -> matrix:
	nrows, ncols = dct_image.shape
	quantized_image = zeros((nrows, ncols), dtype=float32)
	row = 0
	while(row < nrows):
		col = 0
		while(col < ncols):
			quantized_image[row:row+k, col:col+k] = multiply(round(divide(dct_image[row:row+k, col:col+k], q_f_matrix)), q_i_matrix)
			col += k
		row += k
	return quantized_image

# Calcula a quantidade de bits por pixel de uma imagem
def calculate_bpp(image:ndarray) -> int:
	nbytes = 0
	nrows = image.shape[0]
	row = 0
	while row < nrows:
		result = isclose(image[row], 0)
		nbytes = nbytes + count_nonzero(logical_not(result))
		row += 1
	bpp = (nbytes * 8) / (image.shape[0] * image.shape[1])	# Oito (8) é a quantidade de bits que representam cada pixel da imagem
	return bpp

# Função de transformação de uma matriz em uma matriz de potências de dois - Oliveira
def np2_round(quantization_matrix:matrix) -> matrix:
	return power(2, log2(quantization_matrix).round())

# Função de transformação de uma matriz em uma matriz de potências de dois - Brahimi 
def np2_ceil(quantization_matrix:matrix) -> matrix:
	return power(2, ceil(log2(quantization_matrix)))


# DEFINIÇÕES DE CONSTANTES 

# Matriz de amostra para testes
A = array([[127, 123, 125, 120, 126, 123, 127, 128,],
            [142, 135, 144, 143, 140, 145, 142, 140,],
            [128, 126, 128, 122, 125, 125, 122, 129,],
            [132, 144, 144, 139, 140, 149, 140, 142,],
            [128, 124, 128, 126, 127, 120, 128, 129,],
            [133, 142, 141, 141, 143, 140, 146, 138,],
            [124, 127, 128, 129, 121, 128, 129, 128,],
            [134, 143, 140, 139, 136, 140, 138, 141,]], dtype=float)

# Matrizes de transformação
# Matriz de tranformação JPEG
T_ = calculate_matrix_of_transformation(8)

# Ternária padrão
T0 = array([[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, -1, -1, -1],
            [1, 0, 0, -1, -1, 0, 0, 1],
            [1, 0, -1, -1, 1, 1, 0, -1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, 0, 1, -1, 0, 1, -1],
            [0, -1, 1, 0, 0, 1, -1, 0],
            [0, -1, 1, -1, 1, -1, 1, 0]], dtype=float)

# Ternária Brahimi
TB = array([[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, -1, -1],
            [1, 0, 0, -1,- 1, 0, 0, 1],
            [0, 0, -1, 0, 0, 1, 0, 0],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, 0, 0, 0, 0, 1, -1],
            [0, -1, 1, 0, 0, 1, -1, 0],
            [0, 0, 0, -1, 1, 0, 0, 0]], dtype=float)

# Matrizes de quantização
# Matriz de quantização Q0 (JPEG quantisation requires bit-shifts only - Oliveira)
Q0 = array([[16, 11, 10, 16, 24, 40, 51, 61], 
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]], dtype=float)

# Matriz de quantização (Designing Multiplier-Less JPEG Luminance Quantisation Matrix - Brahimi)
QB = array([[20, 17, 18, 19, 22, 36, 36, 31],
    		[19, 17, 20, 22, 24, 40, 23, 40],
      		[20, 22, 24, 28, 37, 53, 50, 54],
			[22, 20, 25, 35, 45, 73, 73, 58],
			[22, 21, 37, 74, 70, 92, 101, 103],
			[24, 43, 50, 64, 100, 104, 120, 92],
			[45, 100, 62, 79, 100, 70, 70, 101],
			[41, 41, 74, 59, 70, 90, 100, 99]], dtype=float)

# Matrizes diagonais
# Matriz diagonal utilizada por Oliveira
S = matrix(diag([1/(pow(8, 1/2)), 1/pow(6,1/2), 1/2, 1/pow(6,1/2), 1/(pow(8, 1/2)), 1/pow(6,1/2), 1/2, 1/pow(6,1/2)])).T
# Matriz diagonal utilizada por Brahime
SB = matrix(diag([1/(pow(8, 1/2)), 1/2, 1/2, 1/(pow(2, 1/2)), 1/(pow(8, 1/2)), 1/2, 1/2, 1/(pow(2, 1/2))])).T

# Elementos da matriz diagonal vetorizados
s = matrix(diag(S))
sb = matrix(diag(SB))

# Matriz ortogonal
C = dot(S, T0)
CB = dot(SB, TB)
# Matriz diagonal 8x8
Z = dot(s.T, s)
ZB = dot(sb.T, sb)

#TODO Fazer uma função que recebe uma matriz de quantização, uma elevação 'el' e um fator de qualidade 'qf'
def map_k_and_el(row_index:int, image_height:int) -> tuple:
	el = row_index/image_height * pi - pi/2
	kprime = arange(8)
	k = clip(0, 7, around(kprime/cos(el))).astype(int)
	return (k, el)


def deSimone_compression(image:ndarray, quality_factor:int, quantization_matrix:ndarray, transformation_matrix) -> ndarray:
	nrows, ncols = image.shape
	compressed_image = zeros(image.shape)
	N = 8
	transformed_image = apply_exact_transform(transformation_matrix, image, N)
	q_matrix = quantization(quality_factor, quantization_matrix)
	for row_index in range(0, nrows, N):
		k, el = map_k_and_el(row_index, nrows)
		Q = []
		for x in k:
			Q.append(q_matrix.T[x])
		auxImage = transformed_image[row_index:row_index+N]
		quantized_image = apply_quantization(Q, auxImage, N)
		compressed_image[row_index:row_index+N] = apply_inverse_transform(transformation_matrix, quantized_image, N)
	return compressed_image

'''main'''
path_images = "test_images/"
file = "sample-ERP.jpg"
full_path = os.path.join(path_images, file)
image = imread(full_path, as_gray=True)
print(image.shape)
n_image = deSimone_compression(image, 15, Q0, TB)
plot.imshow(n_image, cmap='gray', label=file)
plot.show()

pause()