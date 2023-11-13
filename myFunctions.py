import os
import time as t
import numpy as np
from skimage import io
from skimage import metrics


# =============================================================================================================================================================================================
''' Funções auxiliares '''
def get_image(filename:str, path:str="") -> np.ndarray:
	return io.imread(os.path.join(path, filename))

def is_gray_scale(image:np.ndarray) -> bool:
	is_gray = False
	if image.ndim == 2:
		is_gray = True
	return is_gray

def print_np_matrix(matrix:np.ndarray) -> None:
	width, length = matrix.shape
	for row in range(width):
		for col in range(length):
			print(matrix[row][col], end=" ")
		print()
	
def calculate_matrix_of_transformation(k:int) -> np.ndarray:
	row = 0
	alpha = 0.0
	transformation_matrix = np.zeros((k, k), np.float64)
	while(row < k):								
		if row == 0:
			alpha = 1 / (k ** 0.5)
		else:
			alpha = (2 / k) ** 0.5
		col = 0
		while(col < k):
			transformation_matrix[row][col] = alpha * np.cos((np.pi * row * (2 * col + 1)) / (2 * k))
			col += 1
		row += 1
	return transformation_matrix

def calculate_matrix_of_parametric_quantization(k:int, r:int) -> np.ndarray:
	q_matrix = np.zeros((k, k), np.float64)
	i = 0
	while(i < k):
		j = 0
		while(j < k):
			q_matrix[i][j] = 1 + (i + j) * r
			j += 1
		i += 1
	return q_matrix

def calculate_matrix_of_qf_quantization(qf_value:int, quantization_matrix:np.ndarray) -> np.ndarray:
	s = 0.0
	if qf_value < 50:
		s = 5_000 / qf_value
	else:
		s = 200 - (2 * qf_value)
	resulting_matrix = np.floor((s * quantization_matrix + 50) / 100)
	return resulting_matrix

def calculate_values_of_graphics(image:np.ndarray, transformationMatrix:np.ndarray, k:int) -> (list, list, list, list):
	vec_psnr = []
	vec_ssim = []
	vec_bpp = []
	vec_qf = list(range(5, 96, 5))
	for qf in vec_qf:
		vec_qf.append(qf)
		q = calculate_matrix_of_qf_quantization(qf)
		b = apply_direct_transform(transformationMatrix, image, k)
		b_linha = apply_quantization(q, b, k)
		a_linha = apply_inverse_transform(transformationMatrix, b_linha, k)
		vec_psnr.append(metrics.peak_signal_noise_ratio(image, a_linha, data_range=255))
		vec_ssim.append(metrics.structural_similarity(image, a_linha, data_range=255))
		vec_bpp.append(calculate_number_of_bytes_of_image_per_pixels(b_linha))
	return vec_psnr, vec_ssim, vec_bpp, vec_qf

def calculate_average(psnrMatrix:list, ssimMatrix:list, bppMatrix:list) -> (list, list, list, list):
	psnr_average = []
	ssim_average = []
	bpp_average = []
	qf_average = []
	for index_of_qf in range(len(psnrMatrix[0])):
		psnr = 0
		ssim = 0
		bpp = 0
		qf = 0
		for index_of_image in range(len(psnrMatrix)):
			psnr = psnr + psnrMatrix[index_of_image][index_of_qf]
			ssim = ssim + ssimMatrix[index_of_image][index_of_qf]
			bpp = bpp + bppMatrix[index_of_image][index_of_qf]
		psnr = psnr / len(psnrMatrix)
		ssim = ssim / len(ssimMatrix)
		bpp = bpp / len(bppMatrix)
		qf = index_of_qf * 5 + 5
		psnr_average.append(psnr)
		ssim_average.append(ssim)
		bpp_average.append(bpp)
		qf_average.append(qf)
	return psnr_average, ssim_average, bpp_average, qf_average

def print_matrixes(quantizationType:bool, transformationMatrix:np.ndarray, quantizationMatrix:np.ndarray, originalImageMatrix:np.ndarray, dctMatrix:np.ndarray, quantizedDctMatrix:np.ndarray, idctMatrix: np.ndarray, k:int):
	print(f"Matrix de transformação ({transformationMatrix.dtype}):")
	print(f"{transformationMatrix}\n")
	if quantizationType:
		print('Matriz de quantização - Parametrica :', end="")
	else:
		print('Matriz de quantização - Fator de qualidade :', end="")
	print(f"({quantizationMatrix.dtype})")
	print(f"{quantizationMatrix}\n\n--------------------------------------------------\n\n")
	print(f"Matrix A8 ({originalImageMatrix.dtype})")
	print(f"{originalImageMatrix[0:k, 0:k]}\n")
	print(f"Matrix B8 ({dctMatrix.dtype})")
	print(f"{dctMatrix[0:k, 0:k]}\n")
	print(f"Matrix B8_linha ({quantizedDctMatrix.dtype})")
	print(f"{quantizedDctMatrix[0:k, 0:k]}\n")
	print(f"Matrix A8_linha ({idctMatrix.dtype})")
	print(f"{idctMatrix[0:k, 0:k]}\n")

def calculate_number_of_bytes_of_image_per_pixels(image:np.ndarray) -> int:
	nbytes = 0
	nrows = image.shape[0]
	row = 0
	while row < nrows:
		result = np.isclose(image[row], 0)
		nbytes = nbytes + np.count_nonzero(np.logical_not(result))
		row += 1
	bpp = (nbytes * 8) / (image.shape[0] * image.shape[1])	# Oito (8) é a quantidade de bits que representam cada pixel da imagem
	return bpp

def np2(q_matrix:np.ndarray) -> np.ndarray:
	return 2 ** (np.round(np.log2(q_matrix)))

# =============================================================================================================================================================================================
''' Principais matrizes'''
sample_matrix = np.array([[127, 123, 125, 120, 126, 123, 127, 128],
							[142, 135, 144, 143, 140, 145, 142, 140],
							[128, 126, 128, 122, 125, 125, 122, 129],
							[132, 144, 144, 139, 140, 149, 140, 142],
							[128, 124, 128, 126, 127, 120, 128, 129],
							[133, 142, 141, 141, 143, 140, 146, 138],
							[124, 127, 128, 129, 121, 128, 129, 128],
							[134, 143, 140, 139, 136, 140, 138, 141]])

c_matrix = calculate_matrix_of_transformation(8)

qf_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
					[12, 12, 14, 19, 26, 58, 60, 55],
					[14, 13, 16, 24, 40, 57, 69, 56],
					[14, 17, 22, 29, 51, 54, 80, 62],
					[18, 22, 37, 56, 68, 109, 103, 77],
					[24, 35, 55, 64, 81, 104, 113, 92],
					[49, 64, 78, 87, 103, 121, 120, 101],
					[72, 92, 95, 98, 112, 100, 103, 99]])

# Matriz C0 de baixo custo(APROXIMAÇÃO): https://arxiv.org/abs/1402.6034v1 
c_0_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
					 [1, 1, 1, 0, 0, -1, -1, -1],
					 [1, 0, 0, -1, -1, 0, 0, 1],
					 [1, 0, -1, -1, 1, 1, 0, -1],
					 [1, -1, -1, 1, 1, -1, -1, 1],
					 [1, -1, 0, 1, -1, 0, 1, -1],
					 [0, -1, 1, 0, 0, 1, -1, 0],
					 [0, -1, 1, -1, 1, -1, 1, 0]])

s_matrix = np.diag([1/(2*pow(2, 1/2)), 1/pow(6,1/2), 1/2, 1/pow(6,1/2), 1/(2*pow(2, 1/2)), 1/pow(6,1/2), 1/2, 1/pow(6,1/2)])

s_array = np.matrix([1/(2*pow(2, 1/2)), 1/pow(6,1/2), 1/2, 1/pow(6,1/2), 1/(2*pow(2, 1/2)), 1/pow(6,1/2), 1/2, 1/pow(6,1/2)])

c_ortho_matrix = np.dot(s_matrix, c_0_matrix)
# =============================================================================================================================================================================================
''' Funções principais '''

def apply_direct_transform(transformation_matrix:np.ndarray, image_matrix:np.ndarray, k:int) -> np.ndarray:
	nrows, ncols = image_matrix.shape
	dct_image = np.zeros((nrows, ncols), dtype=np.float32)
	row = 0
	while(row < nrows):
		col = 0
		while(col < ncols):
			dct_image[row:row+k, col:col+k] = np.dot(np.dot(transformation_matrix, image_matrix[row:row+k, col:col+k]), transformation_matrix.T)
			col += k
		row += k
	return dct_image

def apply_inverse_transform(transformation_matrix:np.ndarray, quantized_image_matrix:np.ndarray, k:int) -> np.ndarray:
	nrows, ncols = quantized_image_matrix.shape
	idct_image = np.zeros((nrows, ncols), dtype=np.float32)
	row = 0
	while(row < nrows):
		col = 0
		while(col < ncols):
			idct_image[row:row+k, col:col+k] = np.dot(np.dot(transformation_matrix.T, quantized_image_matrix[row:row+k, col:col+k]), transformation_matrix)
			col += k
		row += k
	return np.clip(idct_image, 0, 255)

def apply_quantization(q_matrix:np.ndarray, dct_image:np.ndarray, k:int) -> np.ndarray:
	nrows, ncols = dct_image.shape
	quantized_image = np.zeros((nrows, ncols), dtype=np.float32)
	row = 0
	while(row < nrows):
		col = 0
		while(col < ncols):
			quantized_image[row:row+k, col:col+k] = np.multiply(np.round(np.divide(dct_image[row:row+k, col:col+k], q_matrix)), q_matrix)
			col += k
		row += k
	return quantized_image

def apply_oliveira_quantization(q_f_matrix:np.matrix, q_i_matrix:np.matrix, dct_image:np.ndarray, k:int) -> np.matrix:
	nrows, ncols = dct_image.shape
	quantized_image = np.zeros((nrows, ncols), dtype=np.float32)
	row = 0
	while(row < nrows):
		col = 0
		while(col < ncols):
			quantized_image[row:row+k, col:col+k] = np.multiply(np.round(np.divide(dct_image[row:row+k, col:col+k], q_f_matrix)), q_i_matrix)
			col += k
		row += k
	return quantized_image
