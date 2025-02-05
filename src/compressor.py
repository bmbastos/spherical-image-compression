import utils
import numpy as np

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
