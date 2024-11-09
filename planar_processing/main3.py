from pdb import set_trace as pause
import myFunctions as func
import myPlots as plot

"""
Objetivo: comparar a aplicação entre a DCT com a matriz de quantização, a RDCT com a matriz de quantização,
a DCT com matriz de quantização np2 (potências de 2) e a RDCT com matriz de quantização np2 (potências de 2)
Ou seja,
1) DCT com quantização
2) RDCT com quantização
3) DCT com quantização em potências de 2
4) RDCT com quantização em potências de 2
"""

path = "images/"
filename = "elaine.512.tiff"
image = func.get_image(filename, path)

transformation_matrix = func.calculate_matrix_of_transformation(8)

q_0 = func.qf_matrix
q_0_binary = func.np2(q_0)
psnr_q_dct = []
psnr_q_rdct = []
psnr_qbinary_dct = []
psnr_qbinary_rdct = []
ssim_q_dct = []
ssim_q_rdct = []
ssim_qbinary_dct = []
ssim_qbinary_rdct = []
bpp_q_dct = []
bpp_q_rdct = []
bpp_qbinary_dct = []
bpp_qbinary_rdct = []
qf_values = list(range(5,96,5))

transformed_image_by_dct = func.apply_direct_transform(func.c_matrix, image, 8)
transformed_image_by_rdct = func.apply_direct_transform(func.c_ortho_matrix, image, 8)

for qf in qf_values:
    q = func.calculate_matrix_of_qf_quantization(qf, func.qf_matrix)
    q_binary = func.np2(q)

    image_quantized_by_q_and_transformed_by_dct = func.apply_quantization(q, transformed_image_by_dct, 8)
    image_quantized_by_q_and_transformed_by_rdct = func.apply_quantization(q, transformed_image_by_rdct, 8)
    image_quantized_by_q_binary_and_transformed_by_dct = func.apply_quantization(q_binary, transformed_image_by_dct, 8)
    image_quantized_by_q_binary_and_transformed_by_rdct = func.apply_quantization(q_binary, transformed_image_by_rdct, 8)

    image_inversed_q_and_dct = func.apply_inverse_transform(func.c_matrix, image_quantized_by_q_and_transformed_by_dct, 8)
    psnr_q_dct.append(func.metrics.peak_signal_noise_ratio(image, image_inversed_q_and_dct, data_range=255))
    ssim_q_dct.append(func.metrics.structural_similarity(image, image_inversed_q_and_dct, data_range=255))
    bpp_q_dct.append(func.calculate_number_of_bytes_of_image_per_pixels(image_quantized_by_q_and_transformed_by_dct))

    image_inversed_q_and_rdct = func.apply_inverse_transform(func.c_ortho_matrix, image_quantized_by_q_and_transformed_by_rdct, 8)
    psnr_q_rdct.append(func.metrics.peak_signal_noise_ratio(image, image_inversed_q_and_rdct, data_range=255))
    ssim_q_rdct.append(func.metrics.structural_similarity(image, image_inversed_q_and_rdct, data_range=255))
    bpp_q_rdct.append(func.calculate_number_of_bytes_of_image_per_pixels(image_quantized_by_q_and_transformed_by_rdct))
    
    image_inversed_q_binary_and_dct = func.apply_inverse_transform(func.c_matrix, image_quantized_by_q_binary_and_transformed_by_dct, 8)
    psnr_qbinary_dct.append(func.metrics.peak_signal_noise_ratio(image, image_inversed_q_binary_and_dct, data_range=255))
    ssim_qbinary_dct.append(func.metrics.structural_similarity(image, image_inversed_q_binary_and_dct, data_range=255))
    bpp_qbinary_dct.append(func.calculate_number_of_bytes_of_image_per_pixels(image_quantized_by_q_binary_and_transformed_by_dct))

    image_inversed_q_binary_and_rdct = func.apply_inverse_transform(func.c_ortho_matrix, image_quantized_by_q_binary_and_transformed_by_rdct, 8)
    psnr_qbinary_rdct.append(func.metrics.peak_signal_noise_ratio(image, image_inversed_q_binary_and_rdct, data_range=255))
    ssim_qbinary_rdct.append(func.metrics.structural_similarity(image, image_inversed_q_binary_and_rdct, data_range=255))
    bpp_qbinary_rdct.append(func.calculate_number_of_bytes_of_image_per_pixels(image_quantized_by_q_binary_and_transformed_by_rdct))


fig, axs = plot.plt.subplots(2, 2, label="Analyzes ~Q (bit-shifts only) between DCT and RDCT")
axs[0, 0].grid(True)
axs[0, 0].set_title("PSNR x QF")
axs[0, 0].plot(qf_values, psnr_q_dct, color='red', label="DCT with Q")
axs[0, 0].plot(qf_values, psnr_q_rdct, color='blue', label="RDCT with Q")
axs[0, 0].plot(qf_values, psnr_qbinary_dct, color='red', label="DCT with ~Q", ls='dashed')
axs[0, 0].plot(qf_values, psnr_qbinary_rdct, color='blue', label="RDCT with ~Q", ls='dashed')
axs[0, 0].set_xlabel("QF values")
axs[0, 0].set_ylabel("PSNR values")
axs[0, 0].legend()

axs[0, 1].grid(True)
axs[0, 1].set_title("SSIM x QF")
axs[0, 1].plot(qf_values, ssim_q_dct, color='red', label="DCT with Q")
axs[0, 1].plot(qf_values, ssim_q_rdct, color='blue', label="RDCT with Q")
axs[0, 1].plot(qf_values, ssim_qbinary_dct, color='red', label="DCT with ~Q", ls='dashed')
axs[0, 1].plot(qf_values, ssim_qbinary_rdct, color='blue', label="RDCT with ~Q", ls='dashed')
axs[0, 1].set_xlabel("QF values")
axs[0, 1].set_ylabel("SSIM values")
axs[0, 1].legend()

axs[1, 0].grid(True)
axs[1, 0].set_title("PSNR x BPP")
axs[1, 0].plot(bpp_q_dct, psnr_q_dct, color='red', label="DCT with Q")
axs[1, 0].plot(bpp_q_rdct, psnr_q_rdct, color='blue', label="RDCT with Q")
axs[1, 0].plot(bpp_qbinary_dct, psnr_qbinary_dct, color='red', label="DCT with ~Q", ls='dashed')
axs[1, 0].plot(bpp_qbinary_rdct, psnr_qbinary_rdct, color='blue', label="RDCT with ~Q", ls='dashed')
axs[1, 0].set_xlabel("BPP values")
axs[1, 0].set_ylabel("PSNR values")
axs[1, 0].legend()

axs[1, 1].grid(True)
axs[1, 1].set_title("SSIM x BPP")
axs[1, 1].plot(bpp_q_dct, ssim_q_dct, color='red', label="DCT with Q")
axs[1, 1].plot(bpp_q_rdct, ssim_q_rdct, color='blue', label="RDCT with Q")
axs[1, 1].plot(bpp_qbinary_dct, ssim_qbinary_dct, color='red', label="DCT with ~Q", ls='dashed')
axs[1, 1].plot(bpp_qbinary_rdct, ssim_qbinary_rdct, color='blue', label="RDCT with ~Q", ls='dashed')
axs[1, 1].set_xlabel("BPP values")
axs[1, 1].set_ylabel("SSIM values")
axs[1, 1].legend()

fig.tight_layout()
plot.plt.show()
