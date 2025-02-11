from src.compressor import Compressor
from skimage.io import imread


compressor = Compressor(imread('input_images/AerialCity_3840x1920_30fps_8bit_420_erp_0.bmp'))
compressor.our_methodology()
print(compressor.get_image())