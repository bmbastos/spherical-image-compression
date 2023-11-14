import myFunctions as func
import myPlots as plot
from skimage import color

"""
Objetivo: Implementar a varredura de um conjunto de imagens, aplicar a transformada em cada uma delas
 com a matriz de quantização dependendo de q_factor (variando de 5 à 95) e calcular as médias entre 
 PSNR x QF, SSIM x QF, PSNR x BPP e SSIM x BPP
"""

k = 8															# Dimensão do kernel (k x k)
r = 10															# Nível de compressão da imagem
parametric_quantization = False									# Tipo de quantização (parametrica ou fator de qualidade)
quality_factor = 95												# Fator de qualidade caso seja escolhido o esse tipo de quantização
path = "images/"												# Local de armazenamento das imagens a serem processadas

c = func.calculate_matrix_of_transformation(k)						# Cálculo da matrix de transformação
q = 0.0															    # A partir da variável booleana escolhemos como será feita a quantização
if parametric_quantization:
	q = func.calculate_matrix_of_parametric_quantization(k, r)
else:
	q = func.calculate_matrix_of_qf_quantization(quality_factor, func.qf_matrix)

files = func.os.listdir(path)										# Carregamento dos arquivos localizados em path
n_images = 0

psnr_matrix = []
ssim_matrix = []
bpp_matrix = []
qf_matrix = []

start_time = func.t.time()											# Conta o inicio do tempo de processamento

for file in files:						    						# Processamento das imagens
	image = 0
	print(file)
	if file.endswith('.tiff'):
		full_path = func.os.path.join(path, file)
		image = func.get_image(full_path)
		if not func.is_gray_scale(image):
			image = color.rgb2gray(image)
		psnr, ssim, bpp, qf = func.calculate_values_of_graphics(image, c, k)
		psnr_matrix.append(psnr)
		ssim_matrix.append(ssim)
		bpp_matrix.append(bpp)
		qf_matrix.append(qf)
		n_images += 1

end_time = func.t.time()												    # Conta o fim do tempo de processamento

print(f"Tempo de processamento das {n_images} imagens: {func.np.round(end_time - start_time, 0)} segundos")

media_psnr = 0.0
media_ssim = 0.0
media_bpp = 0.0
media_qf = 0.0
media_psnr, media_ssim, media_bpp, media_qf = func.calculate_average(psnr_matrix, ssim_matrix, bpp_matrix)
image_label = "Average of " + str({len(files)}) + "images"
plot.plot_graphics(media_psnr, media_ssim, media_bpp, media_qf, image_label)

