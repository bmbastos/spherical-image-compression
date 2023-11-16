from pdb import set_trace as pause
from skimage import io, metrics, color
from numpy import *
from matplotlib import pyplot as plt
import myFunctions as func
import os

"""
Objetivo: Implementar a proposta do artigo do Oliveira
 e comparar os sinais de cada uma das aplicações
Feature: Mesclar proposta do(a) Brahime
"""

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Definicoes
# Matriz de amostra para testes
A = array([[127, 123, 125, 120, 126, 123, 127, 128,],
            [142, 135, 144, 143, 140, 145, 142, 140,],
            [128, 126, 128, 122, 125, 125, 122, 129,],
            [132, 144, 144, 139, 140, 149, 140, 142,],
            [128, 124, 128, 126, 127, 120, 128, 129,],
            [133, 142, 141, 141, 143, 140, 146, 138,],
            [124, 127, 128, 129, 121, 128, 129, 128,],
            [134, 143, 140, 139, 136, 140, 138, 141,]], dtype=float)

# Matriz de tranformação padrão
T = func.calculate_matrix_of_transformation(8)

# Matriz de transformação ternária padrão
C_0 = array([[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, -1, -1, -1],
            [1, 0, 0, -1, -1, 0, 0, 1],
            [1, 0, -1, -1, 1, 1, 0, -1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, 0, 1, -1, 0, 1, -1],
            [0, -1, 1, 0, 0, 1, -1, 0],
            [0, -1, 1, -1, 1, -1, 1, 0]], dtype=float)

# Matriz de transformação ternária utilizada por Brahime
TB = array([[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, -1, -1],
            [1, 0, 0, -1,- 1, 0, 0, 1],
            [0, 0, -1, 0, 0, 1, 0, 0],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, 0, 0, 0, 0, 1, -1],
            [0, -1, 1, 0, 0, 1, -1, 0],
            [0, 0, 0, -1, 1, 0, 0, 0]])

# Matriz de quantização QF
Q = array([[16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 54, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]], dtype=float)

# Matriz de quantização utilizada por Brahime
QB = array([[20, 17, 18, 19, 22, 36, 36, 31],
            [19, 17, 20, 22, 24, 40, 23, 40],
            [20, 22, 24, 28, 37, 53, 50, 54],
            [22, 20, 25, 35, 45, 73, 73, 58],
            [22, 21, 37, 74, 70, 92, 101, 103],
            [24, 43, 50, 64, 100, 104, 120, 92],
            [45, 100, 62, 79, 100, 70, 70, 101],
            [41, 41, 74, 59, 70, 90, 100, 99]], dtype=float)

# Matriz diagonal
S = matrix(diag([1/(pow(8, 1/2)), 1/pow(6,1/2), 1/2, 1/pow(6,1/2), 1/(pow(8, 1/2)), 1/pow(6,1/2), 1/2, 1/pow(6,1/2)])).T
# Matriz diagonal utilizada por Brahime
SB = matrix(diag([1/(pow(8, 1/2)), 1/2, 1/2, 1/(pow(2, 1/2)), 1/(pow(8, 1/2)), 1/2, 1/2, 1/(pow(2, 1/2))])).T
# Elementos da matriz diagonal vetorizados
s = matrix(diag(S))
sb = matrix(diag(SB))
# Matriz ortogonal
C = dot(S, C_0)
CB = dot(SB, TB)
# Matriz diagonal 8x8
Z = dot(s.T, s)
ZB = dot(sb.T, sb)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Funções

# Função np2
def np2(q_matrix:matrix) -> matrix:
    return power(2, log2(q_matrix).round())

# Função np2b: Brahime
def np2b(q_matrix:matrix) -> matrix:
    return power(2, ceil(log2(q_matrix)))
            
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Implementação

path = "images/"
files = os.listdir(path)
qt_images = 20
avg_psnr_dct = []
avg_psnr_rdct_op1 = []
avg_psnr_rdct_op2 = []
avg_psnr_brahime = []
avg_ssim_dct = []
avg_ssim_rdct_op1 = []
avg_ssim_rdct_op2 = []
avg_ssim_brahime = []
avg_bpp_dct = []
avg_bpp_rdct_op1 = []
avg_bpp_rdct_op2 = []
avg_bpp_brahime = []
vec_qf = list(range(5, 96, 5))
pause()
psnr_dct = []
ssim_dct = []
bpp_dct = []
psnr_rdct_op1 = []
ssim_rdct_op1 = []
bpp_rdct_op1 = []        
psnr_rdct_op2 = []
ssim_rdct_op2 = []
bpp_rdct_op2 = []
psnr_brahime = []
ssim_brahime = []
bpp_brahime = []

count_images = 0
while count_images < qt_images:
    file = files[count_images]
    if file.endswith('.tiff'):
        full_path = os.path.join(path, file)
        image = io.imread(full_path)
        if not func.is_gray_scale(image):
            image = color.rgb2gray(image)
        
        print(file)

        vec_dct_psnr = []
        vec_dct_ssim = []
        vec_dct_bpp = []
        vec_rdct_oliveira_op1_psnr = []
        vec_rdct_oliveira_op1_ssim = []
        vec_rdct_oliveira_op1_bpp = []
        vec_rdct_oliveira_op2_psnr = []
        vec_rdct_oliveira_op2_ssim = []
        vec_rdct_oliveira_op2_bpp = []
        vec_rdct_brahime_psnr = []
        vec_rdct_brahime_ssim = []
        vec_rdct_brahime_bpp = []


        # Aplicação da transformada simples e aproximada, repectivamente
        image_dct = func.apply_direct_transform(T, image, 8)
        image_rdct_op1 = func.apply_direct_transform(C, image, 8)
        image_rdct_op2 = func.apply_direct_transform(C_0, image, 8)
        image_rdct_brahime = func.apply_direct_transform(TB, image, 8)

        for qf in vec_qf:
            quantization_matrix_qf = func.calculate_matrix_of_qf_quantization(qf, Q)
            quantization_matrix_brahime = func.calculate_matrix_of_qf_quantization(qf, QB)  
            QO_f_matrix = np2(divide(quantization_matrix_qf, Z))
            QO_i_matrix = np2(multiply(Z, quantization_matrix_qf))
            QB_f_matrix = np2b(divide(quantization_matrix_brahime, ZB))
            QB_i_matrix = np2b(multiply(ZB, quantization_matrix_brahime))

            # Aplicação da quantização nas imagens transformadas diretamente
            quantized_image_dct = func.apply_quantization(quantization_matrix_qf, image_dct, 8)
            quantized_image_rdct_op1 = func.apply_quantization(quantization_matrix_qf, image_rdct_op1, 8)
            quantized_image_rdct_op2 = func.apply_oliveira_quantization(QO_f_matrix, QO_i_matrix, image_rdct_op2, 8)
            quantized_image_rdct_brahime = func.apply_oliveira_quantization(QB_f_matrix, QB_i_matrix, image_rdct_brahime, 8)

            # Aplicação da transformada inversa nas imagens quantizadas
            compressed_image_dct = func.apply_inverse_transform(T, quantized_image_dct, 8)
            compressed_image_rdct_op1 = func.apply_inverse_transform(C, quantized_image_rdct_op1, 8)
            compressed_image_rdct_op2 = func.apply_inverse_transform(C_0, quantized_image_rdct_op2, 8)
            compressed_image_rdct_brahime = func.apply_inverse_transform(TB, quantized_image_rdct_brahime, 8)



            # Coleta de informação
            vec_dct_psnr.append(metrics.peak_signal_noise_ratio(image, compressed_image_dct, data_range=255))
            vec_dct_ssim.append(metrics.structural_similarity(image, compressed_image_dct, data_range=255))
            vec_dct_bpp.append(func.calculate_number_of_bytes_of_image_per_pixels(quantized_image_dct))
            vec_rdct_oliveira_op1_psnr.append(metrics.peak_signal_noise_ratio(image, compressed_image_rdct_op1, data_range=255))
            vec_rdct_oliveira_op1_ssim.append(metrics.structural_similarity(image, compressed_image_rdct_op1, data_range=255))
            vec_rdct_oliveira_op1_bpp.append(func.calculate_number_of_bytes_of_image_per_pixels(quantized_image_rdct_op1))
            vec_rdct_oliveira_op2_psnr.append(metrics.peak_signal_noise_ratio(image, compressed_image_rdct_op2, data_range=255))
            vec_rdct_oliveira_op2_ssim.append(metrics.structural_similarity(image, compressed_image_rdct_op2, data_range=255))
            vec_rdct_oliveira_op2_bpp.append(func.calculate_number_of_bytes_of_image_per_pixels(quantized_image_rdct_op2))
            vec_rdct_brahime_psnr.append(metrics.peak_signal_noise_ratio(image, compressed_image_rdct_brahime, data_range=255))
            vec_rdct_brahime_ssim.append(metrics.structural_similarity(image, compressed_image_rdct_brahime, data_range=255))
            vec_rdct_brahime_bpp.append(func.calculate_number_of_bytes_of_image_per_pixels(quantized_image_rdct_brahime))
        
        psnr_dct.append(vec_dct_psnr)
        psnr_rdct_op1.append(vec_rdct_oliveira_op1_psnr)
        psnr_rdct_op2.append(vec_rdct_oliveira_op2_psnr)
        psnr_brahime.append(vec_rdct_brahime_psnr)
        ssim_dct.append(vec_dct_ssim)
        ssim_rdct_op1.append(vec_rdct_oliveira_op1_ssim)
        ssim_rdct_op2.append(vec_rdct_oliveira_op2_ssim)
        ssim_brahime.append(vec_rdct_brahime_ssim)
        bpp_dct.append(vec_dct_bpp)
        bpp_rdct_op1.append(vec_rdct_oliveira_op1_bpp)
        bpp_rdct_op2.append(vec_rdct_oliveira_op2_bpp)
        bpp_brahime.append(vec_rdct_brahime_bpp)
        
    count_images += 1
    ratio = round(count_images / qt_images * 100, 2)
    os.system('clear')
    print(f"Processed images ratio: {ratio}%")

pause()
for index_of_qf in range(len(vec_qf)):
    dct_psnr = 0
    rdct_op1_psnr = 0
    rdct_op2_psnr = 0
    brahime_psnr = 0
    dct_ssim = 0
    rdct_op1_ssim = 0
    rdct_op2_ssim = 0
    brahime_ssim = 0
    dct_bpp = 0
    rdct_op1_bpp = 0
    rdct_op2_bpp = 0
    brahime_bpp = 0
    for index_of_file in range(qt_images):
        dct_psnr += psnr_dct[index_of_file][index_of_qf]
        rdct_op1_psnr += psnr_rdct_op1[index_of_file][index_of_qf]
        rdct_op2_psnr += psnr_rdct_op2[index_of_file][index_of_qf]
        brahime_psnr += psnr_brahime[index_of_file][index_of_qf]
        dct_ssim += ssim_dct[index_of_file][index_of_qf]
        rdct_op1_ssim += ssim_rdct_op1[index_of_file][index_of_qf]
        rdct_op2_ssim += ssim_rdct_op2[index_of_file][index_of_qf]
        brahime_ssim += ssim_brahime[index_of_file][index_of_qf]
        dct_bpp += bpp_dct[index_of_file][index_of_qf]
        rdct_op1_bpp += bpp_rdct_op1[index_of_file][index_of_qf]
        rdct_op2_bpp += bpp_rdct_op2[index_of_file][index_of_qf]
        brahime_bpp += bpp_brahime[index_of_file][index_of_qf]
    avg_psnr_dct.append(dct_psnr/qt_images)
    avg_psnr_rdct_op1.append(rdct_op1_psnr/qt_images)
    avg_psnr_rdct_op2.append(rdct_op2_psnr/qt_images)
    avg_psnr_brahime.append(brahime_psnr/qt_images)
    avg_ssim_dct.append(dct_ssim/qt_images)
    avg_ssim_rdct_op1.append(rdct_op1_ssim/qt_images)
    avg_ssim_rdct_op2.append(rdct_op2_ssim/qt_images)
    avg_ssim_brahime.append(brahime_ssim/qt_images)
    avg_bpp_dct.append(dct_bpp/qt_images)
    avg_bpp_rdct_op1.append(rdct_op1_bpp/qt_images)
    avg_bpp_rdct_op2.append(rdct_op2_bpp/qt_images)
    avg_bpp_brahime.append(brahime_bpp/qt_images)

# Plotagem dos gráficos
fig, axs = plt.subplots(2, 2, label="Averages of 20 images")
axs[0, 0].grid(True)
axs[0, 0].set_title("QF X PSNR")
axs[0, 0].plot(vec_qf, avg_psnr_dct, color='red', label="standard-dct")
axs[0, 0].plot(vec_qf, avg_psnr_rdct_op1, color='green', label="rdct-option1", ls='dashed')
axs[0, 0].plot(vec_qf, avg_psnr_rdct_op2, color='green', label="rdct-option2")
axs[0, 0].plot(vec_qf, avg_psnr_brahime, color='blue', label="Brahime-propose")
axs[0, 0].set_xlabel("QF values")
axs[0, 0].set_ylabel("PSNR values")
axs[0, 0].legend()

axs[0, 1].grid(True)
axs[0, 1].set_title("QF X SSIM")
axs[0, 1].plot(vec_qf, avg_ssim_dct, color='red', label="standard-dct")
axs[0, 1].plot(vec_qf, avg_ssim_rdct_op1, color='green', label="rdct-option1", ls='dashed')
axs[0, 1].plot(vec_qf, avg_ssim_rdct_op2, color='green', label="rdct-option2")
axs[0, 1].plot(vec_qf, avg_ssim_brahime, color='blue', label="Brahime-propose")
axs[0, 1].set_xlabel("QF values")
axs[0, 1].set_ylabel("SSIM values")
axs[0, 1].legend()

axs[1, 0].grid(True)
axs[1, 0].set_title("RD Curve (BPP X PSNR)")
axs[1, 0].plot(avg_bpp_dct, avg_psnr_dct, color='red', label="standard-dct")
axs[1, 0].plot(avg_bpp_rdct_op1, avg_psnr_rdct_op1, color='green', label="rdct-option1", ls='dashed')
axs[1, 0].plot(avg_bpp_rdct_op2, avg_psnr_rdct_op2, color='green', label="rdct-option2")
axs[1, 0].plot(avg_bpp_brahime, avg_psnr_brahime, color='blue', label="Brahime-propose")
axs[1, 0].set_xlabel("BPP")
axs[1, 0].set_ylabel("PSNR values")
axs[1, 0].legend()

axs[1, 1].grid(True)
axs[1, 1].set_title("RD CUrve (BPP X SSIM)")
axs[1, 1].plot(avg_bpp_dct, avg_ssim_dct, color='red', label="standard-dct")
axs[1, 1].plot(avg_bpp_rdct_op1, avg_ssim_rdct_op1, color='green', label="rdct-option1", ls='dashed')
axs[1, 1].plot(avg_bpp_rdct_op2, avg_ssim_rdct_op2, color='green', label="rdct-option2")
axs[1, 1].plot(avg_bpp_brahime, avg_ssim_brahime, color='blue', label="Brahime-propose")
axs[1, 1].set_xlabel("BPP")
axs[1, 1].set_ylabel("SSIM values")
axs[1, 1].legend()

fig.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#Anotações 

# Opcao 1

B = dot(dot(C, A), C.T)
B_til = multiply(Q, around(divide(B, Q)))
A_til = dot(dot(C.T, B_til), C)

# Opcao 2

Q_f = np2(divide(Q, Z))
Q_i = np2(multiply(Z, Q))

TAT = dot(dot(C_0, A), C_0.T)
B_til2 = multiply(around(divide(TAT, Q_f)), Q_i)
A_til2 = dot(dot(C_0.T, B_til2), C_0)

# Teste de sanidade

r = sum(A_til - A_til2)
r = "{:.10f}".format(r)

pause()