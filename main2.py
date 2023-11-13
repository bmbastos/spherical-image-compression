import myFunctions as func
import myPlots as plot

"""
Objetivo: Comparar a DCT com a RDCT
"""

path = "images/"
file_name = "boat.512.tiff"
image = func.get_image(file_name, path)

psnr_values, ssim_values, bpp_values, qf_values = func.calculate_values_of_graphics(image, func.c_matrix, 8)
dct_vectors = [psnr_values, ssim_values, bpp_values, qf_values]

psnr_values, ssim_values, bpp_values, qf_values = func.calculate_values_of_graphics(image, func.c_ortho_matrix, 8)
rdct_vectors = [psnr_values, ssim_values, bpp_values, qf_values]

plot.plot_graphics_of_two_models(dct_vectors, rdct_vectors)

func.print_np_matrix(func.np.round(func.np.dot(func.c_ortho_matrix, func.c_ortho_matrix.T), 2))
