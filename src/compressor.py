from src.matrices import *
from scipy import signal
import numpy as np
from pdb import set_trace as pause

class Compressor:
	
	def __init__(self, image:np.ndarray=None, block_size:int=8, transformation_matrix:np.ndarray=TR, quantization_matrix:np.ndarray=Q0, quantization_factor:int=50, np2:str='Round'):
		""" Initializer  """
		self.image = image
		if image is None:
			raise ValueError("Image is required")
		self.block_size = block_size
		self.quantization_factor = quantization_factor
		self.transformation_matrix = transformation_matrix
		self.quantization_matrix = quantization_matrix
		self.dequantization_matrix = quantization_matrix			
		self.np2 = np2
		
	# Getters and Setters
	def get_image(self):
		return self.image
	
	def get_block_size(self):
		return self.block_size
	
	def get_transformation_matrix(self):
		return self.transformation_matrix
	
	def get_quantization_matrix(self):
		return self.quantization_matrix
	
	def get_dequantization_matrix(self):
		return self.dequantization_matrix
	
	def get_quantization_factor(self):
		return self.quantization_factor
	
	def get_np2(self):
		return self.np2
	
	def set_image(self, image:np.ndarray):
		self.image = image

	def set_block_size(self, block_size:int):
		self.block_size = block_size

	def set_transformation_matrix(self, transformation_matrix:np.ndarray):
		self.transformation_matrix = transformation_matrix

	def set_quantization_matrix(self, quantization_matrix:np.ndarray):
		self.quantization_matrix = quantization_matrix
			
	def set_dequantization_matrix(self, dequantization_matrix:np.ndarray):
		self.dequantization_matrix = dequantization_matrix

	def set_quantization_factor(self, quantization_factor:int):
		if quantization_factor < 1 or quantization_factor > 100:
			raise ValueError("Invalid quantization factor")
		self.quantization_factor = quantization_factor

	def set_np2(self, np2:str):
		if np2 not in ['Round', 'Ceil', 'Floor']:
			raise ValueError("Invalid np2 value")
		self.np2 = np2

	@staticmethod
	def calculate_matrix_of_transformation(k:int) -> np.ndarray:
		""" Calculate the transformation matrix of JPEG with size k """
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
	
	@staticmethod
	def process_image(image:np.ndarray):
		""" Process image """
		if image.max() <= 1:
			image = np.around(255*image)
		return image

	@staticmethod
	def umount(image_data:np.ndarray, block_size:int=8):
		""" Unmounts a matrix into blocks  """
		nrows, ncols = block_size, block_size
		h, w= image_data.shape 
		if h % nrows != 0 or w % ncols != 0:
			raise ValueError(f"The matrix ({h}x{w}) cannot be divided exactly into blocks of {nrows}x{ncols}")
		return image_data.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)

	@staticmethod
	def remount(image_data:np.ndarray, new_dimension:tuple[int, int]):
		""" Remounts a matrix into blocks  """
		h, w = new_dimension
		_, nrows, ncols = image_data.shape
		return image_data.reshape(h//nrows, -1, nrows, ncols).swapaxes(1,2).reshape(h, w)

	@staticmethod
	def map_k_and_el(row_index:int, image_height:int) -> tuple[int, float]:
		""" Calculates the elevation angle (el) of an image line and the indices (k) for distortions """
		el = row_index/image_height * np.pi - np.pi/2
		kprime = np.arange(8)
		k = np.clip(np.around(kprime/np.cos(el)), 0, 7).astype(int)
		return k, el
		
	@staticmethod
	def build_LUT(image_height:int, N:int=8) -> tuple:
		""" Builds the Look-up-table of the image based on its height """
		ks, els = [], []
		for row_index in range(0, image_height+1, N):
			k, el = Compressor.map_k_and_el(row_index, image_height)
			ks.append(k); els.append(el)
		ks = np.asarray(ks); els = abs(np.asarray(els))
		k_LUT = np.unique(ks, axis=0)
		min_LUT = np.asarray([np.finfo('f').max for x in k_LUT])
		max_LUT = []
		aux_max_LUT = [np.finfo('f').min for x in k_LUT]
		for idx in range(len(ks)):
			for idx2 in range(len(k_LUT)):
				if np.sum(ks[idx] - k_LUT[idx2]) == 0:
					if els[idx] > aux_max_LUT[idx2]: 
						aux_max_LUT[idx2] = els[idx]
					if els[idx] < min_LUT[idx2]: 
						min_LUT[idx2] = els[idx] 
		for idx in range(len(k_LUT)):
			if idx != len(k_LUT)-1: 
				max_LUT.append(min_LUT[idx+1])
			else: 
				max_LUT.append(aux_max_LUT[idx])
		max_LUT = np.asarray(max_LUT)
		return k_LUT, min_LUT, max_LUT
	
	@staticmethod
	def QtildeAtEl(quantization_matrix:np.ndarray, k_lut:np.ndarray, min_lut:np.ndarray, max_lut:np.ndarray, el:np.float32):
		""" Returns the quantization matrix for a given elevation angle """
		ks = None
		Q = []
		el = abs(el) # LUT is mirrored
		for idx in range(len(k_lut)):
			if el >= min_lut[idx] and el < max_lut[idx]: 
				ks = k_lut[idx]
		if ks is None and np.isclose(el, 0): ks = k_lut[0]
		for k in ks:
			Q.append(quantization_matrix.T.tolist()[k])
		Q = np.asarray(Q)
		Q = Q.T
		return Q
	
	def __scale_quantization(self) -> np.ndarray:
		""" Adjust quantization matrix by a scalar factor """
		s = 5000.0 / self.quantization_factor if self.quantization_factor < 50 else 200.0 - 2.0 * self.quantization_factor
		self.quantization_matrix = np.floor((s * self.quantization_matrix + 50) / 100)
		self.dequantization_matrix = self.quantization_matrix.copy()
	
	def __compute_quantization_dequantization_matrices(self) -> tuple[np.matrix, np.matrix]:
		""" Compute quantization and dequantization matrices through the transformation matrix """
		adjustment_matrix = np.matrix(np.sqrt(np.clip(np.linalg.inv(np.dot(self.transformation_matrix, self.transformation_matrix.T)), 0, None)))
		adjustment_vector = np.matrix(np.diag(adjustment_matrix))
		Z = np.dot(adjustment_vector.T, adjustment_vector)
		if len(self.quantization_matrix.shape) == 3:
			Z = np.tile(np.asarray([Z]), (self.image.shape[0], 1, 1))
		self.quantization_matrix = np.array(np.divide(self.quantization_matrix, Z))
		self.dequantization_matrix =  np.array(np.multiply(self.dequantization_matrix,  Z))
		#self.dequantization_matrix =  np.array(np.multiply(Z, self.dequantization_matrix))
	
	def __prepareQPhi(self, original_image_shape:tuple):
		""" Builds the Look-up-table of the image based on its height """
		h, w = original_image_shape
		k_lut, min_lut, max_lut = Compressor.build_LUT(h)
		els = np.linspace(-np.pi/2, np.pi/2, h//self.block_size+1)
		els = 0.5*(els[1:] + els[:-1]) # gets the "central" block elevation
		QPhiForward, QPhiBackward = [], []
		for el in els: 
			QPhiForward.append(Compressor.QtildeAtEl(self.quantization_matrix, k_lut, min_lut, max_lut, el))
			QPhiBackward.append(Compressor.QtildeAtEl(self.dequantization_matrix, k_lut, min_lut, max_lut, el))
		QPhiForward = np.asarray(QPhiForward)
		QPhiBackward = np.asarray(QPhiBackward)
		self.quantization_matrix = np.repeat(QPhiForward, w//self.block_size, axis=0)
		self.dequantization_matrix = np.repeat(QPhiBackward, w//self.block_size, axis=0)

	def __direct_transform(self):
		""" 2D DCT """
		self.image = np.einsum('mij, jk -> mik', np.einsum('ij, mjk -> mik', self.transformation_matrix, self.image), self.transformation_matrix.T)
		
	def __quantize(self):
		""" Quantization forward """
		self.image = np.around(np.divide(self.image, self.quantization_matrix))

	def __dequantize(self):
		""" Quantization backward """
		self.image = np.multiply(self.image, self.dequantization_matrix)

	def __inverse_transform(self):
		""" 2D IDCT """
		self.image = np.einsum('mij, jk -> mik', np.einsum('ij, mjk -> mik', self.transformation_matrix.T, self.image), self.transformation_matrix)

	def __np2_round(self) -> np.ndarray:
		""" Approximation function for powers of two based on the rounding function - Oliveira (2016) """
		self.quantization_matrix = np.power(2, np.around(np.log2(self.quantization_matrix)))
		self.dequantization_matrix = np.power(2, np.around(np.log2(self.dequantization_matrix)))

	def __np2_ceil(self) -> np.ndarray:# Don't use this
		""" Approximation function for powers of two based on the ceiling function - Brahimi (2021) """
		self.quantization_matrix = np.power(2, np.ceil(np.log2(self.quantization_matrix)))
		self.dequantization_matrix = np.power(2, np.ceil(np.log2(self.dequantization_matrix)))

	def __np2_floor(self) -> np.ndarray: # Don't use this
		""" Approximation function for powers of two based on the flooring function - Us (2025) """
		self.quantization_matrix = np.power(2, np.floor(np.log2(self.quantization_matrix)))
		self.dequantization_matrix = np.power(2, np.floor(np.log2(self.dequantization_matrix)))

	def calculate_bpp(compressed_image:np.ndarray):
		""" Calculate bits per pixel """
		#print("Calculating BPP...")
		return np.count_nonzero(np.logical_not(np.isclose(compressed_image, 0))) * 8 / (compressed_image.shape[0] * compressed_image.shape[1])

	@staticmethod
	def calculate_wpsnr(original_image:np.ndarray, target_image:np.ndarray, max_value:int=255):
		""" Calculate Weighted-to-Spherically-Uniform PSNR """
		def __weights(height, width):
			phis = np.arange(height+1) * np.pi / height
			deltaTheta = 2 * np.pi / width 
			column = np.asarray([deltaTheta * (-np.cos(phis[j+1])+np.cos(phis[j])) for j in range(height)])
			return np.repeat(column[:, np.newaxis], width, 1)
		
		#print("Calculating WS-PSNR...")

		w = __weights(original_image.shape[0], original_image.shape[1])
		wmse = np.sum((original_image - target_image)**2 * w) / (4 * np.pi)
		return float(10 * np.log10(max_value**2 / wmse))

	@staticmethod
	def calculate_wssim(original_image:np.ndarray, target_image:np.ndarray, K1:float=0.01, K2:float=0.03, L:int=255):
		""" Calculate Weighted-to-Spherically-Uniform SSIM """
		def __fspecial_gauss(size:int, sigma:float):
			""" Function to mimic fspecial('gaussian',...) from MATLAB """
			x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
			g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
			return g / g.sum()
		
		def __weights(height, width):
			deltaTheta = 2 * np.pi / width 
			column = np.asarray([np.cos( deltaTheta * (j - height/2.+0.5)) for j in range(height)])
			return np.repeat(column[:, np.newaxis], width, 1)
		
		#print("Calculating WS-SSIM...")
		
		original_image = np.float64(original_image)
		target_image = np.float64(target_image)

		k = 11
		sigma = 1.5
		window = __fspecial_gauss(k, sigma)
		window2 = np.zeros_like(window); window2[k//2, k//2] = 1

		C1 = (K1 * L)**2
		C2 = (K2 * L)**2

		mu1 = signal.convolve2d(original_image, window, 'valid')
		mu2 = signal.convolve2d(target_image, window, 'valid')

		mu1_sq = mu1*mu1
		mu2_sq = mu2*mu2
		mu1_mu2 = mu1*mu2

		sigma1_sq = signal.convolve2d(original_image*original_image, window, 'valid') - mu1_sq
		sigma2_sq = signal.convolve2d(target_image*target_image, window, 'valid') - mu2_sq
		sigma12 = signal.convolve2d(original_image*target_image, window, 'valid') - mu1_mu2

		W = __weights(*original_image.shape)
		Wi = signal.convolve2d(W, window2, 'valid')

		ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)) * Wi
		return float(np.sum(ssim_map)/np.sum(Wi))


	def lower_complexity(self, original_image:np.ndarray):
		""" The base proposal of our algorithm.  """
		self.set_transformation_matrix(TR)
		self.set_quantization_matrix(Q0)
		self.set_dequantization_matrix(Q0)
		self.image = Compressor.umount(self.image, self.block_size)
		self.__direct_transform()
		self.__scale_quantization()
		self.__compute_quantization_dequantization_matrices()
		self.__np2_round()
		self.__prepareQPhi(original_image.shape)
		self.__quantize()

		compressed_image = self.image.reshape(original_image.shape[0], original_image.shape[1])

		self.__dequantize()
		self.__inverse_transform()
		self.image = np.clip(Compressor.remount(self.image, (original_image.shape[0], original_image.shape[1])), 0, 255)
		#print("Image compressed successfully")
		return compressed_image

	def lower_complexity_2(self, original_image:np.ndarray):
		""" The proposal equivalent to lower_complexity. """
		self.set_transformation_matrix(TR)
		self.set_quantization_matrix(Q0)
		self.set_dequantization_matrix(Q0)
		self.image = Compressor.umount(self.image, self.block_size)
		self.__direct_transform()
		self.__scale_quantization()
		self.__compute_quantization_dequantization_matrices()
		self.__prepareQPhi(original_image.shape)	# The only difference is that we use QPhi first, then we apply the np2_round
		self.__np2_round()
		self.__quantize()

		compressed_image = self.image.reshape(original_image.shape[0], original_image.shape[1])

		self.__dequantize()
		self.__inverse_transform()
		self.image = np.clip(Compressor.remount(self.image, (original_image.shape[0], original_image.shape[1])), 0, 255)
		#print("Image compressed successfully")
		return compressed_image

	
	def matematically_correct(self, original_image:np.ndarray):
		""" The proposal that obtains the bests results, but isn't the most efficient. """
		self.set_transformation_matrix(TR)
		self.set_quantization_matrix(Q0)
		self.set_dequantization_matrix(Q0)
		self.image = Compressor.umount(self.image, self.block_size)
		self.__direct_transform()
		self.__scale_quantization()
		self.__prepareQPhi(original_image.shape)
		self.__compute_quantization_dequantization_matrices()
		self.__np2_round()
		self.__quantize()

		compressed_image = self.image.reshape(original_image.shape[0], original_image.shape[1])

		self.__dequantize()
		self.__inverse_transform()
		self.image = np.clip(Compressor.remount(self.image, (original_image.shape[0], original_image.shape[1])), 0, 255)
		#print("Image compressed successfully")
		return compressed_image


	def proposed_from_oliveira(self, original_image):
		""" Original proposal from Oliveira (2016) """
		self.set_transformation_matrix(TO)
		self.set_quantization_matrix(Q0)
		self.set_dequantization_matrix(Q0)
		#print("Compressing image...")
		self.image = Compressor.umount(self.image, self.block_size)
		self.__direct_transform()
		self.__scale_quantization()
		self.__compute_quantization_dequantization_matrices()
		self.__np2_round()
		self.__quantize()

		compressed_image = self.image.reshape(original_image.shape[0], original_image.shape[1])

		self.__dequantize()
		self.__inverse_transform()
		self.image = np.clip(Compressor.remount(self.image, (original_image.shape[0], original_image.shape[1])), 0, 255)
		#print("Image compressed successfully")
		return compressed_image


	def proposed_from_brahimi(self, original_image):
		""" Original proposal from Brahimi (2021) """
		self.set_transformation_matrix(TB)
		self.set_quantization_matrix(QB)
		self.set_dequantization_matrix(QB)
		#print("Compressing image...")
		self.image = Compressor.umount(self.image, self.block_size)
		self.__direct_transform()
		self.__scale_quantization()
		self.__compute_quantization_dequantization_matrices()
		#if self.quantization_factor > 50 and self.quantization_factor < 65: pause()
		self.__np2_ceil()
		#if self.quantization_factor > 50 and self.quantization_factor < 65: pause()
		self.__quantize()

		compressed_image = self.image.reshape(original_image.shape[0], original_image.shape[1])

		self.__dequantize()
		#print("Pós dequantize")
		#print(f"Max = {np.max(self.image)}")
		#print(f"Min = {np.min(self.image)}")
		
		self.__inverse_transform()
		self.image = np.clip(Compressor.remount(self.image, (original_image.shape[0], original_image.shape[1])), 0, 255)
		#print("Image compressed successfully")
		return compressed_image
	

	def proposed_from_raiza(self, original_image):
		""" Original Proposed from Raiza (2018) """
		self.set_transformation_matrix(TR)
		self.set_quantization_matrix(Q0)
		self.set_dequantization_matrix(Q0)
		#print("Compressing image...")
		self.image = Compressor.umount(self.image, self.block_size)
		self.__direct_transform()
		self.__scale_quantization()
		self.__compute_quantization_dequantization_matrices()
		self.__quantize()

		compressed_image = self.image.reshape(original_image.shape[0], original_image.shape[1])

		self.__dequantize()
		self.__inverse_transform()
		self.image = np.clip(Compressor.remount(self.image, (original_image.shape[0], original_image.shape[1])), 0, 255)
		#print("Image compressed successfully")
		return compressed_image
	

	def proposed_from_araar(self, original_image):
		""" Original Proposed from Araar (2022) """
		self.set_transformation_matrix(Compressor.calculate_matrix_of_transformation(self.block_size))
		self.set_quantization_matrix(QHVS)
		self.set_dequantization_matrix(QHVS)
		#print("Compressing image...")
		self.image = Compressor.umount(self.image, self.block_size)
		self.__direct_transform()
		self.__scale_quantization()
		self.__compute_quantization_dequantization_matrices()
		self.__np2_round()
		self.__quantize()

		compressed_image = self.image.reshape(original_image.shape[0], original_image.shape[1])

		self.__dequantize()
		self.__inverse_transform()
		self.image = np.clip(Compressor.remount(self.image, (original_image.shape[0], original_image.shape[1])), 0, 255)
		#print("Image compressed successfully")
		return compressed_image
	
	def proposed_from_simone(self, original_image):
		""" Original Proposed from Simone (2016) """
		self.set_transformation_matrix(Compressor.calculate_matrix_of_transformation(self.block_size))
		self.set_quantization_matrix(Q0)
		self.set_dequantization_matrix(Q0)
		#print("Compressing image...")
		self.image = Compressor.umount(self.image, self.block_size)
		self.__direct_transform()
		self.__scale_quantization()
		self.__prepareQPhi(original_image.shape)
		self.__quantize()

		compressed_image = self.image.reshape(original_image.shape[0], original_image.shape[1])

		self.__dequantize()
		self.__inverse_transform()
		self.image = np.clip(Compressor.remount(self.image, (original_image.shape[0], original_image.shape[1])), 0, 255)
		#print("Image compressed successfully")
		return compressed_image
	
	def standard_jpeg(self, original_image):
		""" Original JPEG """
		self.set_transformation_matrix(Compressor.calculate_matrix_of_transformation(self.block_size))
		self.set_quantization_matrix(Q0)
		self.set_dequantization_matrix(Q0)
		#print("Compressing image...")
		self.image = Compressor.umount(self.image, self.block_size)
		self.__direct_transform()
		self.__scale_quantization()
		self.__quantize()

		compressed_image = self.image.reshape(original_image.shape[0], original_image.shape[1])

		self.__dequantize()
		self.__inverse_transform()
		self.image = np.clip(Compressor.remount(self.image, (original_image.shape[0], original_image.shape[1])), 0, 255)
		#print("Image compressed successfully")
		return compressed_image