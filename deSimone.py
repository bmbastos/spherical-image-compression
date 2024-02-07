import os
from numpy import *
from time import time
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from matplotlib import pyplot as plot
from pdb import set_trace as pause
import tools

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

# Função que calcula a quantidade de bits por pixels
def calculate_number_of_bytes_of_image_per_pixels(image:ndarray) -> int:
	nbytes = 0
	nrows = image.shape[0]
	row = 0
	while row < nrows:
		result = isclose(image[row], 0)
		nbytes = nbytes + count_nonzero(logical_not(result))
		row += 1
	bpp = (nbytes * 8) / (image.shape[0] * image.shape[1])	# Oito (8) é a quantidade de bits que representam cada pixel da imagem
	return bpp


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
T = calculate_matrix_of_transformation(8)

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


#TODO Fazer uma função que recebe uma matriz de quantização, uma elevação 'el' e um fator de qualidade 'qf'
def map_k_and_el(row_index:int, image_height:int) -> tuple:
	el = row_index/image_height * pi - pi/2
	kprime = arange(8)
	k = clip(0, 7, around(kprime/cos(el))).astype(int)
	return (k, el)

# Constroi a Look-up-table da imagem com base em sua altura
def build_LUT(image_height:int, N:int=8) -> (ndarray, ndarray, ndarray):
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
		QM = quantization(QF, quantization_matrix)
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
	A = tools.Tools.umount(image, (N, N))# - 128
	Aprime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', transformation_matrix, A), transformation_matrix.T) # forward transform
	Aprime2 = multiply(quantization_matrix, around(divide(Aprime1, quantization_matrix))) # quantization
	Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', transformation_matrix.T, Aprime2), transformation_matrix) # inverse transform
	B = tools.Tools.remount(Aprime3, (h, w)) #+ 128
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
	A = tools.Tools.umount(image, (N, N))# - 128
	Aprime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', transformation_matrix, A), transformation_matrix.T) # forward transform
	Aprime2 = multiply(divide(Aprime1, quantization_matrix_forward).round(), quantization_matrix_backward) # quantization
	Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', transformation_matrix.T, Aprime2), transformation_matrix) # inverse transform
	B = tools.Tools.remount(Aprime3, (h, w)) #+ 128
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

def use_q_phi(image:ndarray, q_matrix: ndarray, qf:int, block_len:int, response:bool) -> ndarray:
	if response == True:
		return prepareQPhi(image, q_matrix, qf, block_len)
	else:
		q = []
		for _ in range(round(image.shape[0] * image.shape[1] / 8 / 8)):
			q.append(quantization(qf, q_matrix))
		q = asarray(q)
		return q

def use_aproximation_transform(transformation_matrix:ndarray, diagonal_matrix:ndarray, response:bool) -> ndarray:
	"""
	Utiliza a matriz de ajuste para o cálculo da matriz de transformação.
	"""
	if response == True:
		return dot(diagonal_matrix, transformation_matrix)
	else:
		return transformation_matrix
	
def use_adjustment_on_quantization(q_matrix, diag_matrix, use_qfit):
	if use_qfit:
		return dot(q_matrix)

def deSimone_compression_low_complexity(image, t_matrix, q_matrix, qf, use_qphi, use_qfit, use_np2, np2_type):
	"""
	Aplica a compressão de imagem de baixo custo
	param image: ndarray    -> imagem a ser processada
	param t_matrix: ndarray -> matriz de transformação
	param q_matrix: ndarray -> matriz de quantização
	param qf: int			-> fator de qualidade da compressão
	param use_qphi: bool	-> aplica o método da De Simone
	param use_afit: bool	-> aplica o ajuste no passo de quantização
	param use_np2:  bool	-> aplica o arredondamento em potências de 2
	param np2_type: str		-> identifica qual o tipo de arredondamento de np2
	"""
	h, w = image.shape
	N = 8
	S, s = compute_scale_matrix(t_matrix)
	Z = dot(s.T, s)
	q_forward, q_backward = encodeQuantiseNDecodeBrahimiB(image, q_matrix, Z, use_np2, np2_type, N) # Fazer uma função que retorna duas matrizes de quantização (Q_forward e Q_backward) e bpp_aux
	QPhi_forward = use_q_phi(image, q_forward, qf, N, use_qphi)
	QPhi_backward = use_q_phi(image, q_backward, qf, N, use_qphi)
	A = tools.Tools.umount(image, (N, N))# - 128
	Aprime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', t_matrix, A), t_matrix.T) # forward transform
	Aprime2 = multiply(divide(Aprime1, QPhi_forward).round(), QPhi_backward) # quantization
	Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', t_matrix.T, Aprime2), t_matrix) # inverse transform
	B = tools.Tools.remount(Aprime3, (h, w)) #+ 128
	return Aprime2.reshape(h,w), clip(B, 0, 255) 

def deSimone_compression_low_complexity(image, t_matrix, q_matrix, qf, use_qphi, use_np2, np2_type):
	"""
	Aplica a compressão de imagem de baixo custo
	param image: ndarray    -> imagem a ser processada
	param t_matrix: ndarray -> matriz de transformação
	param q_matrix: ndarray -> matriz de quantização
	param qf: int			-> fator de qualidade da compressão
	param use_qphi: bool	-> aplica o método da De Simone
	param use_afit: bool	-> aplica o ajuste no passo de quantização
	param use_np2:  bool	-> aplica o arredondamento em potências de 2
	param np2_type: str		-> identifica qual o tipo de arredondamento de np2
	"""
	h, w = image.shape
	N = 8
	S, s = compute_scale_matrix(t_matrix)
	Z = dot(s.T, s)
	q_forward, q_backward = encodeQuantiseNDecodeBrahimiB(q_matrix, Z) # Fazer uma função que retorna duas matrizes de quantização (Q_forward e Q_backward) e bpp_aux
	np2_q_forward = copy(q_forward)
	np2_q_backward = copy(q_backward)
	if use_np2:
		if np2_type == 'O':
			np2_q_forward = np2_round(q_forward)
			np2_q_backward = np2_round(q_backward)
		else:
			np2_q_forward = np2_ceil(q_forward)
			np2_q_backward = np2_ceil(q_backward)
	QPhi_forward = use_q_phi(image, np2_q_forward, qf, N, use_qphi)
	QPhi_backward = use_q_phi(image, np2_q_backward, qf, N, use_qphi)
	A = tools.Tools.umount(image, (N, N))# - 128
	Aprime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', t_matrix, A), t_matrix.T) # forward transform
	Aprime2 = multiply(divide(Aprime1, QPhi_forward).round(), QPhi_backward) # quantization
	Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', t_matrix.T, Aprime2), t_matrix) # inverse transform
	B = tools.Tools.remount(Aprime3, (h, w)) #+ 128
	B = clip(B, 0, 255)
	if flag:
		print(f"\n\nValue of QF: {qf}")
		print(f"\nVetor diagonal: {s}")
		print(f"\nMatriz diagonal: {S}")
		print(f"\nMatriz Q_forward:\n{q_forward}")
		print(f"\nMatriz Q_backward:\n{q_backward}")
		print(f"\nMatriz NP2(Q_forward):\n{np2_q_forward}")
		print(f"\nMatriz NP2(Q_backward):\n{np2_q_backward}")
		print(f"\nMatriz QPhi_forward:\n{QPhi_forward}")
		print(f"\nMatriz QPhi_backward:\n{QPhi_backward}")
		print(f"\nMatriz R:\n{B}")
	return Aprime2.reshape(h,w), B

def deSimone_compression(image:ndarray, q_phi:bool=False, aproximation:bool=False, brahimi_propose:bool=False, apply_np2:bool=False, np2_type:str='O', transformation_matrix:ndarray=None, quantization_matrix:ndarray=None, quality_factor: int = 50) -> ndarray:
	if transformation_matrix.any == None or quantization_matrix.any == None:
		print("Erro: É necessário fornecer as matrizes de transformação e quantização ")
		exit()
	else:
		N = 8												
		
		S, s = compute_scale_matrix(transformation_matrix)
		Z = dot(s.T, s)

		Q_ = use_q_phi(image, quantization_matrix, quality_factor, N, q_phi)
		T_ = use_aproximation_transform(transformation_matrix, S, aproximation)

		bpp_aux, image_r = encodeQuantiseNDecodeBrahimi(image, T_, Q_, Z, apply_np2, np2_type) if brahimi_propose else encodeQuantiseNDecode(image, T_, Q_)

		image_r = clip(image_r, 0, 255)
		return bpp_aux, image_r

'''main'''
path_images = "test_images/"
file = "sample-ERP.jpg"
full_path = os.path.join(path_images, file)
image = around(255*imread(full_path, as_gray=True))

datas = {}
datas.update({'option1':{'Title': '[Ĉ, Q]', 'QPhi': False, 'AproximateTransformation': True, 'BrahimeQuantization': False , 'NP2': False, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'r', 'lineStyle': 'solid'}})
datas.update({'option2':{'Title': '[T, Qf, Qb]' ,'QPhi': False, 'AproximateTransformation': False, 'BrahimeQuantization': True, 'NP2': False, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'b', 'lineStyle': 'dashed'}})
datas.update({'option3':{'Title': '[Ĉ, QPhi]' ,'QPhi': True, 'AproximateTransformation': True, 'BrahimeQuantization': False, 'NP2': False, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'r', 'lineStyle': 'solid'}})
datas.update({'option4':{'Title': '[T, QPhif, QPhib]' ,'QPhi': True, 'AproximateTransformation': False, 'BrahimeQuantization': True, 'NP2': False, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'blue', 'lineStyle': 'dashed'}})

datas.update({'option5':{'Title': '[Ĉ, Q]', 'QPhi': False, 'AproximateTransformation': True, 'BrahimeQuantization': False, 'NP2': False, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'r', 'lineStyle': 'solid'}})
datas.update({'option6':{'Title': '[T, NP2O(Qf), NP2O(Qb)]', 'QPhi': False, 'AproximateTransformation': False, 'BrahimeQuantization': True, 'NP2': True, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'b', 'lineStyle': 'dashed'}})
datas.update({'option7':{'Title': '[Ĉ, QPhi]', 'QPhi': True, 'AproximateTransformation': True, 'BrahimeQuantization': False, 'NP2': False, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'r', 'lineStyle': 'solid'}})
datas.update({'option8':{'Title': '[T, NP2O(QPhif), NP2O(QPhib)]', 'QPhi': True, 'AproximateTransformation': False, 'BrahimeQuantization': True, 'NP2': True, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'b', 'lineStyle': 'dashed'}})
datas.update({'option9':{'Title': '[T, NP2B(QBf), NP2B(QBb)]', 'QPhi': False, 'AproximateTransformation': False, 'BrahimeQuantization': True, 'NP2': True, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'g', 'lineStyle': 'dashed'}})
datas.update({'option10':{'Title': '[T, NP2B(QBPhif), NP2B(QBPhib)]', 'QPhi': True, 'AproximateTransformation': False, 'BrahimeQuantization': True, 'NP2': True, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'g', 'lineStyle': 'dashed'}})

datas.update({'option11':{'Title': '[Low-complex1]', 'QPhi': False, 'AproximateTransformation': False, 'BrahimeQuantization': True, 'NP2': True, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'black', 'lineStyle': 'dashed'}})
datas.update({'option12':{'Title': '[Low-complex2]', 'QPhi': True, 'AproximateTransformation': False, 'BrahimeQuantization': True, 'NP2': True, 'PSNR': [], 'SSIM': [], 'BPP': [], 'color': 'black', 'lineStyle': 'dashed'}})
# for data in datas:
	# print(f"{data} - \tQPhi:{datas[data]['QPhi']} \tAproximateTransformation:{datas[data]['AproximateTransformation']} \tBrahimeQuantization:{datas[data]['BrahimeQuantization']} \tNP2:{datas[data]['NP2']}")

flag = 0

quality_factors = list(range(5,96,5))
for QF in quality_factors:
	bpp1, n_image1 = deSimone_compression(image, datas['option1']['QPhi'], datas['option1']['AproximateTransformation'], datas['option1']['BrahimeQuantization'], datas['option1']['NP2'], 'O', T0, Q0, QF)
	datas['option1']['PSNR'].append(peak_signal_noise_ratio(image, n_image1, data_range=255))
	datas['option1']['SSIM'].append(structural_similarity(image, n_image1, data_range=255))
	datas['option1']['BPP'].append(count_nonzero(logical_not(isclose(bpp1, 0))) * 8 / (bpp1.shape[0] * bpp1.shape[1]))

	bpp2, n_image2 = deSimone_compression(image, datas['option2']['QPhi'], datas['option2']['AproximateTransformation'], datas['option2']['BrahimeQuantization'], datas['option2']['NP2'], 'O', T0, Q0, QF)
	datas['option2']['PSNR'].append(peak_signal_noise_ratio(image, n_image2, data_range=255))
	datas['option2']['SSIM'].append(structural_similarity(image, n_image2, data_range=255))
	datas['option2']['BPP'].append(count_nonzero(logical_not(isclose(bpp2, 0))) * 8 / (bpp2.shape[0] * bpp2.shape[1]))

	bpp3, n_image3 = deSimone_compression(image, datas['option3']['QPhi'], datas['option3']['AproximateTransformation'], datas['option3']['BrahimeQuantization'], datas['option3']['NP2'], 'O', T0, Q0, QF)
	datas['option3']['PSNR'].append(peak_signal_noise_ratio(image, n_image3, data_range=255))
	datas['option3']['SSIM'].append(structural_similarity(image, n_image3, data_range=255))
	datas['option3']['BPP'].append(count_nonzero(logical_not(isclose(bpp3, 0))) * 8 / (bpp3.shape[0] * bpp3.shape[1]))

	bpp4, n_image4 = deSimone_compression(image, datas['option4']['QPhi'], datas['option4']['AproximateTransformation'], datas['option4']['BrahimeQuantization'], datas['option4']['NP2'], 'O', T0, Q0, QF)
	datas['option4']['PSNR'].append(peak_signal_noise_ratio(image, n_image4, data_range=255))
	datas['option4']['SSIM'].append(structural_similarity(image, n_image4, data_range=255))
	datas['option4']['BPP'].append(count_nonzero(logical_not(isclose(bpp4, 0))) * 8 / (bpp4.shape[0] * bpp4.shape[1]))

	bpp5, n_image5 = deSimone_compression(image, datas['option5']['QPhi'], datas['option5']['AproximateTransformation'], datas['option5']['BrahimeQuantization'], datas['option5']['NP2'], 'O', T0, Q0, QF)
	datas['option5']['PSNR'].append(peak_signal_noise_ratio(image, n_image5, data_range=255))
	datas['option5']['SSIM'].append(structural_similarity(image, n_image5, data_range=255))
	datas['option5']['BPP'].append(count_nonzero(logical_not(isclose(bpp5, 0))) * 8 / (bpp5.shape[0] * bpp5.shape[1]))

	bpp6, n_image6 = deSimone_compression(image, datas['option6']['QPhi'], datas['option6']['AproximateTransformation'], datas['option6']['BrahimeQuantization'], datas['option6']['NP2'], 'O', T0, Q0, QF)
	datas['option6']['PSNR'].append(peak_signal_noise_ratio(image, n_image6, data_range=255))
	datas['option6']['SSIM'].append(structural_similarity(image, n_image6, data_range=255))
	datas['option6']['BPP'].append(count_nonzero(logical_not(isclose(bpp6, 0))) * 8 / (bpp6.shape[0] * bpp6.shape[1]))

	bpp7, n_image7 = deSimone_compression(image, datas['option7']['QPhi'], datas['option7']['AproximateTransformation'], datas['option7']['BrahimeQuantization'], datas['option7']['NP2'], 'O', T0, Q0, QF)
	datas['option7']['PSNR'].append(peak_signal_noise_ratio(image, n_image7, data_range=255))
	datas['option7']['SSIM'].append(structural_similarity(image, n_image7, data_range=255))
	datas['option7']['BPP'].append(count_nonzero(logical_not(isclose(bpp7, 0))) * 8 / (bpp7.shape[0] * bpp7.shape[1]))

	bpp8, n_image8 = deSimone_compression(image, datas['option8']['QPhi'], datas['option8']['AproximateTransformation'], datas['option8']['BrahimeQuantization'], datas['option8']['NP2'], 'O', T0, Q0, QF)
	datas['option8']['PSNR'].append(peak_signal_noise_ratio(image, n_image8, data_range=255))
	datas['option8']['SSIM'].append(structural_similarity(image, n_image8, data_range=255))
	datas['option8']['BPP'].append(count_nonzero(logical_not(isclose(bpp8, 0))) * 8 / (bpp8.shape[0] * bpp8.shape[1]))

	bpp9, n_image9 = deSimone_compression(image, datas['option9']['QPhi'], datas['option9']['AproximateTransformation'], datas['option9']['BrahimeQuantization'], datas['option9']['NP2'], 'B', TB, QB, QF)
	datas['option9']['PSNR'].append(peak_signal_noise_ratio(image, n_image9, data_range=255))
	datas['option9']['SSIM'].append(structural_similarity(image, n_image9, data_range=255))
	datas['option9']['BPP'].append(count_nonzero(logical_not(isclose(bpp9, 0))) * 8 / (bpp9.shape[0] * bpp9.shape[1]))
	
	bpp10, n_image10 = deSimone_compression(image, datas['option10']['QPhi'], datas['option10']['AproximateTransformation'], datas['option10']['BrahimeQuantization'], datas['option10']['NP2'], 'B', TB, QB, QF)
	datas['option10']['PSNR'].append(peak_signal_noise_ratio(image, n_image10, data_range=255))
	datas['option10']['SSIM'].append(structural_similarity(image, n_image10, data_range=255))
	datas['option10']['BPP'].append(count_nonzero(logical_not(isclose(bpp10, 0))) * 8 / (bpp10.shape[0] * bpp10.shape[1]))

	if QF == 50: flag = 1
	bpp11, n_image11 = deSimone_compression_low_complexity(image, T0, Q0, QF, datas['option11']['QPhi'], datas['option11']['NP2'], 'O')
	datas['option11']['PSNR'].append(peak_signal_noise_ratio(image, n_image11, data_range=255))
	datas['option11']['SSIM'].append(structural_similarity(image, n_image11, data_range=255))
	datas['option11']['BPP'].append(count_nonzero(logical_not(isclose(bpp11, 0))) * 8 / (bpp11.shape[0] * bpp11.shape[1]))
	if flag: print(); print(70 * " ="); print()
	bpp12, n_image12 = deSimone_compression_low_complexity(image, T0, Q0, QF, datas['option12']['QPhi'], datas['option12']['NP2'], 'O')
	datas['option12']['PSNR'].append(peak_signal_noise_ratio(image, n_image12, data_range=255))
	datas['option12']['SSIM'].append(structural_similarity(image, n_image12, data_range=255))
	datas['option12']['BPP'].append(count_nonzero(logical_not(isclose(bpp12, 0))) * 8 / (bpp12.shape[0] * bpp12.shape[1]))
	flag = 0

for teste in range(4):
	print("TESTE " + str(teste + 1))
	fig_title = ""
	data1 = "option" + str(2 * teste + 1)
	data2 = "option" + str(2 * teste + 2)
	print(data1)
	print(data2)
	if teste == 0:
		fig_title = "[Ĉ, Q] Vs. [T, Qf, Qb]"
	elif teste == 1:
		fig_title = "[Ĉ, QPhi] Vs. [T, QPhif, QPhib]"
	elif teste == 2:
		fig_title = "[Ĉ, Q] Vs. [T, NP2O(Qf), NP2O(Qb)] Vs. [T, NP2B(QBf), NP2B(QBb)]"
		print('option9')
		print('option11')
	else:
		fig_title = "[Ĉ, QPhi] Vs. [T, NP2O(QPhif), NP2O(QPhib)] Vs. [T, NP2B(QBPhif), NP2B(QBPhib)]"
		print('option10')
		print('option12')
	print()
	fig, axes = plot.subplots(2, 2,label="Teste " + str(teste+1) + " - " + fig_title)
	# Primeiro quadrante
	axes[0, 0].grid(True)
	axes[0, 0].set_title("QF Vs. PSNR")
	axes[0, 0].set_xlabel("QF values")
	axes[0, 0].set_ylabel("PSNR values")
	axes[0, 0].plot(quality_factors, datas[data1]['PSNR'], color=datas[data1]['color'], label=datas[data1]['Title'], ls=datas[data1]['lineStyle'])
	axes[0, 0].plot(quality_factors, datas[data2]['PSNR'], color=datas[data2]['color'], label=datas[data2]['Title'], ls=datas[data2]['lineStyle'])
	if teste == 2:
		axes[0, 0].plot(quality_factors, datas['option9']['PSNR'], color=datas['option9']['color'], label=datas['option9']['Title'], ls=datas['option9']['lineStyle'])
		axes[0, 0].plot(quality_factors, datas['option11']['PSNR'], color=datas['option11']['color'], label=datas['option11']['Title'], ls=datas['option11']['lineStyle'])
	if teste == 3:
		axes[0, 0].plot(quality_factors, datas['option10']['PSNR'], color=datas['option10']['color'], label=datas['option10']['Title'], ls=datas['option10']['lineStyle'])
		axes[0, 0].plot(quality_factors, datas['option12']['PSNR'], color=datas['option12']['color'], label=datas['option12']['Title'], ls=datas['option12']['lineStyle'])
	axes[0, 0].legend()
	# Segundo quadrante
	axes[0, 1].grid(True)
	axes[0, 1].set_title("QF Vs. SSIM")
	axes[0, 1].set_xlabel("QF values")
	axes[0, 1].set_ylabel("SSIM values")
	axes[0, 1].plot(quality_factors, datas[data1]['SSIM'], color=datas[data1]['color'], label=datas[data1]['Title'], ls=datas[data1]['lineStyle'])
	axes[0, 1].plot(quality_factors, datas[data2]['SSIM'], color=datas[data2]['color'], label=datas[data2]['Title'], ls=datas[data2]['lineStyle'])
	if teste == 2:
		axes[0, 1].plot(quality_factors, datas['option9']['SSIM'], color=datas['option9']['color'], label=datas['option9']['Title'], ls=datas['option9']['lineStyle'])
		axes[0, 1].plot(quality_factors, datas['option11']['SSIM'], color=datas['option11']['color'], label=datas['option11']['Title'], ls=datas['option11']['lineStyle'])
	if teste == 3:
		axes[0, 1].plot(quality_factors, datas['option10']['SSIM'], color=datas['option10']['color'], label=datas['option10']['Title'], ls=datas['option10']['lineStyle'])
		axes[0, 1].plot(quality_factors, datas['option12']['SSIM'], color=datas['option12']['color'], label=datas['option12']['Title'], ls=datas['option12']['lineStyle'])
	axes[0, 1].legend()
	# Terceiro quadrante
	axes[1, 0].grid(True)
	axes[1, 0].set_title("BPP Vs. PSNR")
	axes[1, 0].set_xlabel("BPP values")
	axes[1, 0].set_ylabel("PSNR values")
	axes[1, 0].plot(datas[data1]['BPP'], datas[data1]['PSNR'], color=datas[data1]['color'], label=datas[data1]['Title'], ls=datas[data1]['lineStyle'])
	axes[1, 0].plot(datas[data2]['BPP'], datas[data2]['PSNR'], color=datas[data2]['color'], label=datas[data2]['Title'], ls=datas[data2]['lineStyle'])
	if teste == 2:
		axes[1, 0].plot(datas['option9']['BPP'], datas['option9']['PSNR'], color=datas['option9']['color'], label=datas['option9']['Title'], ls=datas['option9']['lineStyle'])
		axes[1, 0].plot(datas['option11']['BPP'], datas['option11']['PSNR'], color=datas['option11']['color'], label=datas['option11']['Title'], ls=datas['option11']['lineStyle'])
	if teste == 3:
		axes[1, 0].plot(datas['option10']['BPP'], datas['option10']['PSNR'], color=datas['option10']['color'], label=datas['option10']['Title'], ls=datas['option10']['lineStyle'])
		axes[1, 0].plot(datas['option12']['BPP'], datas['option12']['PSNR'], color=datas['option12']['color'], label=datas['option12']['Title'], ls=datas['option12']['lineStyle'])
	axes[1, 0].legend()
	# Quarto quadrante
	axes[1, 1].grid(True)
	axes[1, 1].set_title("BPP Vs. SSIM")
	axes[1, 1].set_xlabel("BPP values")
	axes[1, 1].set_ylabel("SSIM values")
	axes[1, 1].plot(datas[data1]['BPP'], datas[data1]['SSIM'], color=datas[data1]['color'], label=datas[data1]['Title'], ls=datas[data1]['lineStyle'])
	axes[1, 1].plot(datas[data2]['BPP'], datas[data2]['SSIM'], color=datas[data2]['color'], label=datas[data2]['Title'], ls=datas[data2]['lineStyle'])
	if teste == 2:
		axes[1, 1].plot(datas['option9']['BPP'], datas['option9']['SSIM'], color=datas['option9']['color'], label=datas['option9']['Title'], ls=datas['option9']['lineStyle'])
		axes[1, 1].plot(datas['option11']['BPP'], datas['option11']['SSIM'], color=datas['option11']['color'], label=datas['option11']['Title'], ls=datas['option11']['lineStyle'])
	if teste == 3:
		axes[1, 1].plot(datas['option10']['BPP'], datas['option10']['SSIM'], color=datas['option10']['color'], label=datas['option10']['Title'], ls=datas['option10']['lineStyle'])
		axes[1, 1].plot(datas['option12']['BPP'], datas['option12']['SSIM'], color=datas['option12']['color'], label=datas['option12']['Title'], ls=datas['option12']['lineStyle'])
	axes[1, 1].legend()
	fig.tight_layout()
	plot.show()

