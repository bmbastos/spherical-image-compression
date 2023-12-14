import myFunctions as func
from matplotlib import pyplot as plt

def print_image(originalImage:func.np.ndarray) -> None:
	plt.imshow(originalImage, cmap='gray', label="Image")
	plt.show()

def plot_graphics(psnr_average_vec:list, ssim_average_vec:list, bpp_average_vec:list, qf_vec:list, label:str) -> None:
	fig, axs = plt.subplots(2, 2, label=label)
	axs[0, 0].grid(True)
	axs[0, 0].set_title("QF X PSNR")
	axs[0, 0].plot(qf_vec, psnr_average_vec, color='red')
	axs[0, 0].set_xlabel("QF values")
	axs[0, 0].set_ylabel("PSNR values")

	axs[0, 1].grid(True)
	axs[0, 1].set_title("QF X SSIM")
	axs[0, 1].plot(qf_vec, ssim_average_vec, color='green')
	axs[0, 1].set_xlabel("QF values")
	axs[0, 1].set_ylabel("SSIM values")
	
	axs[1, 0].grid(True)
	axs[1, 0].set_title("RD Curve (BPP X PSNR)")
	axs[1, 0].plot(bpp_average_vec, psnr_average_vec, color='red')
	axs[1, 0].set_xlabel("BPP")
	axs[1, 0].set_ylabel("PSNR values")
	
	axs[1, 1].grid(True)
	axs[1, 1].set_title("RD CUrve (BPP X SSIM)")
	axs[1, 1].plot(bpp_average_vec, ssim_average_vec, color='green')
	axs[1, 1].set_xlabel("BPP")
	axs[1, 1].set_ylabel("SSIM values")
	fig.tight_layout()
	plt.show()

def plot_graphics_of_two_models(first_model:list, second_model:list) -> None:
	fig, axs = plt.subplots(2, 2, label="Differance of RD-Curve between DCT and RDCT")
	axs[0, 0].grid(True)
	axs[0, 0].set_title("QF X PSNR")
	axs[0, 0].plot(first_model[3], first_model[0], color='red', label="DCT")
	axs[0, 0].plot(second_model[3], second_model[0], color='blue', label="RDCT")
	axs[0, 0].set_xlabel("QF values")
	axs[0, 0].set_ylabel("PSNR values")
	axs[0, 0].legend()

	axs[0, 1].grid(True)
	axs[0, 1].set_title("QF X SSIM")
	axs[0, 1].plot(first_model[3], first_model[1], color='red', label="DCT")
	axs[0, 1].plot(second_model[3], second_model[1], color='blue', label="RDCT")
	axs[0, 1].set_xlabel("QF values")
	axs[0, 1].set_ylabel("SSIM values")
	axs[0, 1].legend()

	axs[1, 0].grid(True)
	axs[1, 0].set_title("RD Curve (BPP X PSNR)")
	axs[1, 0].plot(first_model[2], first_model[0], color='red', label="DCT")
	axs[1, 0].plot(second_model[2], second_model[0], color='blue', label="RDCT")
	axs[1, 0].set_xlabel("BPP")
	axs[1, 0].set_ylabel("PSNR values")
	axs[1, 0].legend()

	axs[1, 1].grid(True)
	axs[1, 1].set_title("RD CUrve (BPP X SSIM)")
	axs[1, 1].plot(first_model[2], first_model[1], color='red', label="DCT")
	axs[1, 1].plot(second_model[2], second_model[1], color='blue', label="RDCT")
	axs[1, 1].set_xlabel("BPP")
	axs[1, 1].set_ylabel("SSIM values")
	axs[1, 1].legend()
	fig.tight_layout()
	plt.show()

def plot_images(originalImage:func.np.ndarray, dctImage:func.np.ndarray, quantizedImage:func.np.ndarray, idctImage:func.np.ndarray, parametric_quantization:bool) -> None:
	fig, axes = plt.subplots(2, 2, label="Images")
	ax = axes.ravel()
	ax[0].imshow(originalImage, cmap='gray')
	ax[0].set_title("Original")
	ax[1].imshow(dctImage, cmap='gray')
	ax[1].set_title("DCT", )
	ax[2].imshow(quantizedImage, cmap='gray')
	if parametric_quantization:
		ax[2].set_title("Quantized - Parametric", )
	else: 
		ax[2].set_title("Quantized - QF", )
	ax[3].imshow(idctImage, cmap='gray')
	ax[3].set_title("IDCT")
	plt.show()
	
