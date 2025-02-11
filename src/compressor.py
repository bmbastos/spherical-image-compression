from src.matrices import *
import numpy as np

class Compressor:
	def __init__(self, image:np.ndarray=None, block_size:int=8, transformation_matrix:np.ndarray=TR, quantization_matrix:np.ndarray=Q0, quantization_factor:int=50, np2:str='Round'):
		""" Compress image """
		if image is None:
			raise ValueError("Image is required")
		self.image = image
		self.block_size = block_size
		self.transformation_matrix = transformation_matrix
		self.quantization_matrix = quantization_matrix
		self.dequantization_matrix = None
		self.quantization_factor = quantization_factor
		self.np2 = np2
		self.wpsnr = None
		self.wssim = None
		self.bpp = None

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
	
	def get_wpsnr(self):
		return self.wpsnr
	
	def get_wssim(self):
		return self.wssim
	
	def get_bpp(self):
		return self.bpp
	
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
		self.quantization_factor = quantization_factor

	def set_np2(self, np2:str):
		self.np2 = np2



	def umount(data:np.ndarray, newdimension:tuple=(8, 8)):
		if len(newdimension) == 1:
			n = newdimension[0]
			return [data[i:i + n] for i in range(0, len(data), n)]
		elif len(newdimension) == 2:
			nrows, ncols = newdimension
			h, w = data.shape
			if h % nrows != 0 or w % ncols != 0:
				raise ValueError(f"The matrix ({h}x{w}) cannot be divided exactly into blocks of {nrows}x{ncols}")
			return (data.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

	def remount(data:np.ndarray, newdimension:tuple=(8, 8)):
		if len(newdimension) == 1:
			return [value for chunk in data for value in chunk]
		elif len(newdimension) == 2:
			h, w = newdimension
			_, nrows, ncols = data.shape
			return (data.reshape(h//nrows, -1, nrows, ncols).swapaxes(1,2).reshape(h, w))
		
	def compute_adjustment_matrix(transformation_matrix:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		""" Compute adjustment matrix obtained by the transformation matrix """
		adjustment_matrix = np.array(np.sqrt(np.linalg.inv(np.dot(transformation_matrix, transformation_matrix.T)))) 
		adjustment_vector = np.array(np.diag(adjustment_matrix))
		return adjustment_matrix, adjustment_vector
	
	def scale_quantization(quality_factor:int, quantization_matrix:np.ndarray) -> np.ndarray:
		""" Adjust quantization matrix by a scalar factor """
		s = 5000.0 / quality_factor if quality_factor < 50 else 200.0 - 2.0 * quality_factor
		return np.floor((s * quantization_matrix + 50) / 100)
	
	def compute_quantization_dequantization_matrices(self, transformation_matrix:np.ndarray, quantization_matrix:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		""" Compute quantization and dequantization matrices """
		S, s = self.compute_adjustment_matrix(transformation_matrix)
		Z = np.dot(s.T, s)
		Q_forward = np.divide(quantization_matrix, Z)
		Q_backward =  np.multiply(quantization_matrix,  Z)
		return Q_forward, Q_backward
	
	def map_k_and_el(row_index:int, image_height:int) -> tuple:
		el = row_index/image_height * np.pi - np.pi/2
		kprime = np.arange(8)
		k = np.clip(0, 7, np.around(kprime/np.cos(el))).astype(int)
		return (k, el)

	def build_LUT(self, image_height:int, N:int=8) -> tuple:
		ks, els = [], []
		for row_index in range(0, image_height+1, N):
			k, el = self.map_k_and_el(row_index, image_height)
			ks.append(k); els.append(el)
		ks = np.asarray(ks); els = abs(np.asarray(els))
		k_LUT = np.unique(ks, axis=0)
		min_LUT = np.asarray([np.finfo('f').max for x in k_LUT])
		max_LUT = []
		aux_max_LUT = [np.finfo('f').min for x in k_LUT]
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
		max_LUT = np.asarray(max_LUT)
		return k_LUT, min_LUT, max_LUT

	def QtildeAtEl(k_lut:np.ndarray, min_lut:np.ndarray, max_lut:np.ndarray, el:np.float32, quantization_matrix:np.ndarray, QF:int= 50):
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
	
	def prepareQPhi(self, image:np.ndarray, quantization_matrix:np.ndarray, N = 8):
		""" Builds the Look-up-table of the image based on its height """
		h, w = image.shape
		k_lut, min_lut, max_lut = self.build_LUT(h)
		els = np.linspace(-np.pi/2, np.pi/2, h//N+1)
		els = 0.5*(els[1:] + els[:-1]) # gets the "central" block elevation
		QPhi = []
		for el in els: 
			QPhi.append(self.QtildeAtEl(k_lut, min_lut, max_lut, el, quantization_matrix))
		QPhi = np.asarray(QPhi)
		QPhi = np.repeat(QPhi, w//N, axis=0)
		return QPhi

	def direct_transform(self, block_size:int):
		""" 2D DCT """
		self.image = np.einsum('mij, jk -> mik', np.einsum('ij, mjk -> mik', self.transformation_matrix, self.umount(self.image, (block_size, block_size))), self.transformation_matrix.T)
		
	def quantize(self):
		""" Quantization forward """
		self.image = np.around(np.divide(self.image, self.quantization_matrix))

	def dequantize(self):
		""" Quantization backward """
		self.image = np.multiply(self.image, self.dequantization_matrix)

	def inverse_transform(self, block_size:int):
		""" 2D IDCT """
		self.image = np.einsum('mij, jk -> mik', np.einsum('ij, mjk -> mik', self.transformation_matrix.T, self.image), self.transformation_matrix)

	def np2_round(quantization_matrix:np.ndarray) -> np.ndarray:
		return np.power(2, np.around(np.log2(quantization_matrix)))
	""" Approximation function for powers of two based on the rounding function - Oliveira (2016) """

	def np2_ceil(quantization_matrix:np.ndarray) -> np.ndarray:
		return np.power(2, np.ceil(np.log2(quantization_matrix))) # Don't use this
	""" Approximation function for powers of two based on the ceiling function - Brahimi (2021) """

	def np2_floor(quantization_matrix:np.ndarray) -> np.ndarray:
		return np.power(2, np.floor(np.log2(quantization_matrix))) # Don't use this
	""" Approximation function for powers of two based on the flooring function - Us (2025) """

	@classmethod
	def our_methodology(self):
		""" Compress image """
		print("Compressing image...")
		