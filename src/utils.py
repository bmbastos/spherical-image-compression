from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import math
import os

A_ = np.array([[127, 123, 125, 120, 126, 123, 127, 128,],
			[142, 135, 144, 143, 140, 145, 142, 140,],
			[128, 126, 128, 122, 125, 125, 122, 129,],
			[132, 144, 144, 139, 140, 149, 140, 142,],
			[128, 124, 128, 126, 127, 120, 128, 129,],
			[133, 142, 141, 141, 143, 140, 146, 138,],
			[124, 127, 128, 129, 121, 128, 129, 128,],
			[134, 143, 140, 139, 136, 140, 138, 141,]], dtype=float)
""" Sample Matrix """

# Low-cost transformation matrices
TO = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
				[1, 1, 1, 0, 0, -1, -1, -1],
				[1, 0, 0, -1, -1, 0, 0, 1],
				[1, 0, -1, -1, 1, 1, 0, -1],
				[1, -1, -1, 1, 1, -1, -1, 1],
				[1, -1, 0, 1, -1, 0, 1, -1],
				[0, -1, 1, 0, 0, 1, -1, 0],
				[0, -1, 1, -1, 1, -1, 1, 0]], dtype=float)
""" Ternary matrix proposed by Cintra (2011) """

TR = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
				[2, 2, 1, 0, 0, -1, -2, -2],
				[2, 1, -1, -2, -2, -1, 1, 2],
				[1, 0, -2, -2, 2, 2, 0, -1],
				[1, -1, -1, 1, 1, -1, -1, 1],
				[2, -2, 0, 1, -1, 0, 2, -2],
				[1, -2, 2, -1, -1, 2, -2, 1],
				[0, -1, 2, -2, 2, -2, 1, 0]], dtype=float)
""" Ternary matrix proposed by Raiza (2018) """

TB = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
				[1, 1, 0, 0, 0, 0, -1, -1],
				[1, 0, 0, -1,- 1, 0, 0, 1],
				[0, 0, -1, 0, 0, 1, 0, 0],
				[1, -1, -1, 1, 1, -1, -1, 1],
				[1, -1, 0, 0, 0, 0, 1, -1],
				[0, -1, 1, 0, 0, 1, -1, 0],
				[0, 0, 0, -1, 1, 0, 0, 0]], dtype=float)
""" Ternary matrix proposed by Brahimi (2020) """

# Quantization Matrices (Bases)
Q0 = np.array([[16, 11, 10, 16, 24, 40, 51, 61], 
				[12, 12, 14, 19, 26, 58, 60, 55],
				[14, 13, 16, 24, 40, 57, 69, 56],
				[14, 17, 22, 29, 51, 87, 80, 62],
				[18, 22, 37, 56, 68, 109, 103, 77],
				[24, 35, 55, 64, 81, 104, 113, 92],
				[49, 64, 78, 87, 103, 121, 120, 101],
				[72, 92, 95, 98, 112, 100, 103, 99]], dtype=float)
""" JPEG Quantization Matrix (1992) """

QB = np.array([[20, 17, 18, 19, 22, 36, 36, 31],
				[19, 17, 20, 22, 24, 40, 23, 40],
				[20, 22, 24, 28, 37, 53, 50, 54],
				[22, 20, 25, 35, 45, 73, 73, 58],
				[22, 21, 37, 74, 70, 92, 101, 103],
				[24, 43, 50, 64, 100, 104, 120, 92],
				[45, 100, 62, 79, 100, 70, 70, 101],
				[41, 41, 74, 59, 70, 90, 100, 99]], dtype=float)	
""" Quantization Matrix proposed by Brahimi (2021) """

QHVS = np.array([[16, 16, 16, 16, 17, 18, 21, 24],
					[16, 16, 16, 16, 17, 19, 22, 25],
					[16, 16, 17, 18, 20, 22, 25, 29],
					[16, 16, 18, 21, 24, 27, 31, 36],
					[17, 17, 20, 24, 30, 35, 41, 47],
					[18, 19, 22, 27, 35, 44, 54, 65],
					[21, 22, 25, 31, 41, 54, 70, 88],
					[24, 25, 29, 36, 47, 65, 88, 115]], dtype=float)
""" Quantization matrix proposed by Araar and Chabbi (2023) """

class Image:
	def __init__(self, image_path:str):
		""" Load image from file """
		if not image_path.exists() or not image_path.is_file():
			raise ValueError(f"The flie {image_path} does not exist or is not a valid file.")
		
		self.data = imread(image_path, as_gray=True).astype(float)
		self.shape = self.data.shape
		self.title = image_path.split('/')[-1]

	def get_image(self):
		""" Return image data """
		return self.data
	
	def get_shape(self):
		""" Return image shape """
		return self.shape
	
	def set_title(self, title:str):
		""" Set image title """
		while not title.endswith('.jpeg'):
			title = input("Please, enter a valid title for the image (e.g. 'image.jpeg'): ")
			if title in os.listdir('./output'):
				change = input("The file already exists. Do you want to replace it? (y/n): ")
				if change == 'y':
					break					
		self.title = title
	
	def show(self):
		""" Show image """
		plt.imshow(self.data, cmap='gray')
		plt.show()

	def save(self):
		""" Save image """
		plt.imsave('./output' + self.title  , self.data, cmap='gray')

class Compressor:
	def __init__(self, image:np.array, block_size=8, transformation_matrix=TR, quantization_matrix=Q0, quantization_factor=50, np2='Round'):
		""" Compress image """
		self.image = image

		def umount(data, newdimension=(8, 8)):
			if len(newdimension) == 1:
				n = newdimension[0]
				return [data[i:i + n] for i in range(0, len(data), n)]
			elif len(newdimension) == 2:
				nrows, ncols = newdimension
				h, w = data.shape
				if h % nrows != 0 or w % ncols != 0:
					raise ValueError(f"The matrix ({h}x{w}) cannot be divided exactly into blocks of {nrows}x{ncols}")
				return (data.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))
		
		def direct_transform(self, block_size, transformation_matrix):
			""" 2D DCT """
			return np.einsum('mij, jk -> mik', np.einsum('ij, mjk -> mik', transformation_matrix, umount(self.image, (block_size, block_size))), transformation_matrix.T)
		
		def quantize(self, )

		
			





# Auxiliary functions