import matplotlib.pyplot as plt
from pdb import set_trace as pause
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import svd
from matrixes import *
from numpy import *

def adjust_quantization(quality_factor:int, quantization_matrix:ndarray) -> ndarray:
	s = 0.0
	if quality_factor < 50:
		s = 5_000 / quality_factor
	else:
		s = 200 - (2 * quality_factor)
	resulting_matrix = floor((s * quantization_matrix + 50) / 100)
	return resulting_matrix
""" Calcula a matriz de quantização dado um fator de quantização """

def compute_scale_matrix(transformation_matrix:ndarray) -> matrix:
	scale_matrix = matrix(sqrt(linalg.inv(dot(transformation_matrix, transformation_matrix.T))))
	scale_vector = matrix(diag(scale_matrix))
	return scale_matrix, scale_vector
""" Matrix diagonal e elementos da matriz diagonal vetorizados """

def np2_round(quantization_matrix:matrix) -> matrix:
	return power(2, around(log2(quantization_matrix)))
""" Função que calcula as potencias de dois mais próximas de uma dada matriz - Oliveira """

def np2_ceil(quantization_matrix:matrix) -> matrix:
	return power(2, ceil(log2(quantization_matrix)))
""" Função que calcula as potencias de dois mais próximas de uma dada matriz - Brahimi """

def np2_floor(quantization_matrix:matrix) -> matrix:
	return power(2, floor(log2(quantization_matrix)))
""" Função que calcula as potencias de dois mais próximas de uma dada matriz - ????? """


""" Main"""
quality_factors = list(range(5, 100, 5))

# Determinante
det_oliveira = []
det_brahimi = []

for qf in quality_factors:
	det_oliveira.append(linalg.det(adjust_quantization(qf, Q0)))
	det_brahimi.append(linalg.det(adjust_quantization(qf, QB)))

# Traço
trace_oliveira = []
trace_brahimi = []

for qf in quality_factors:
	trace_oliveira.append(sum(diag(adjust_quantization(qf, Q0))))
	trace_brahimi.append(sum(diag(adjust_quantization(qf, QB))))

# Rank(Posto)
rank_oliveira = []
rank_brahimi = []

for qf in quality_factors:
	rank_oliveira.append(linalg.matrix_rank(adjust_quantization(qf, Q0)))
	rank_brahimi.append(linalg.matrix_rank(adjust_quantization(qf, QB)))

# Norma
norm_oliveira = []
norm_brahimi = []

for qf in quality_factors:
	norm_oliveira.append(linalg.matrix_norm(adjust_quantization(qf, Q0)))
	norm_brahimi.append(linalg.matrix_norm(adjust_quantization(qf, QB)))

# Condicionamento********
cond_oliveira = []
cond_brahimi = []

for qf in quality_factors:
	cond_oliveira.append(linalg.cond(adjust_quantization(qf, Q0)))
	cond_brahimi.append(linalg.cond(adjust_quantization(qf, QB)))

# Ortogonalidade
orto_oliveira = []
orto_brahimi = []

for qf in quality_factors:
	oliveira_inv = linalg.inv(adjust_quantization(qf, Q0))
	brahimi_inv = linalg.inv(adjust_quantization(qf, QB))
	orto_oliveira.append(allclose(adjust_quantization(qf, Q0).T, oliveira_inv))
	orto_brahimi.append(allclose(adjust_quantization(qf, QB).T, brahimi_inv))

# Decomposição SVD
for qf in quality_factors:
	[S_oliveira, V_oliveira, D_oliveira] = svd(adjust_quantization(qf, Q0))
	[S_brahimi, V_brahimi, D_brahimi] = svd(adjust_quantization(qf, QB))
	"""
	print(f"QF: {qf}")
	print(f"S de Oliveira:\n{S_oliveira}")
	print(f"S de Brahimi:\n{S_brahimi}\n")
	print(f"V de Oliveira:\n{V_oliveira}")
	print(f"V de Brahimi:\n{V_brahimi}\n")
	print(f"D de Oliveira:\n{D_oliveira}")
	print(f"D de Brahimi:\n{D_brahimi}\n")
	pause()"""

# Decomposição QR
for qf in quality_factors:
	Q_oliveira, R_oliveira = linalg.qr(adjust_quantization(qf, Q0))
	Q_brahimi, R_brahimi = linalg.qr(adjust_quantization(qf, QB))
	"""
	print(f"Q de Oliveira:\n{Q_oliveira}")
	print(f"Q de Brahimi:\n{Q_brahimi}\n")
	print(f"R de Oliveira:\n{R_oliveira}")
	print(f"R de Brahimi:\n{R_brahimi}\n")
	print(f"QF: {qf}")
	pause()"""

# Autovalores e autovetores
for qf in quality_factors:
	w_oliveira, v_oliveira = linalg.eig(adjust_quantization(qf, Q0))
	w_brahimi, v_brahimi = linalg.eig(adjust_quantization(qf, QB))
	"""
	print(f"Autovalores de Oliveira:\n{w_oliveira}")
	print(f"Autovalores de Brahimi:\n{w_brahimi}\n")
	print(f"Autovetores de Oliveira:\n{v_oliveira}")
	print(f"Autovetores de Brahimi:\n{v_brahimi}\n")
	print(f"QF: {qf}")
	pause()
	"""

oliveira_elements = (Q0.flatten().astype(int).tolist())
brahimi_elements = (QB.flatten().astype(int).tolist())
print(f"{sorted(oliveira_elements)}")
print(f"{sorted(brahimi_elements)}\n")

print(f"Oliveira Soma = {sum(oliveira_elements)}")
print(f"Brahimi Soma = {sum(brahimi_elements)}")

print(f"Oliveira Média = {mean(oliveira_elements)}")
print(f"Brahimi Média = {mean(brahimi_elements)}\n")

print("Simetria diagonal:")
print(allclose(Q0, Q0.T))
print(allclose(QB, QB.T))

print("Simetria central:")
print(allclose(Q0, flip(Q0)))
print(allclose(QB, flip(QB)))




low_freq_value_oliveira = Q0[0, 0]  # valor no canto superior esquerdo
high_freq_value_oliveira = Q0[-1, -1]  # valor no canto inferior direito
low_freq_value_brahimi = QB[0, 0]  # valor no canto superior esquerdo
high_freq_value_brahimi = QB[-1, -1]  # valor no canto inferior direito

gradient_ratio_oliveira = high_freq_value_oliveira / low_freq_value_oliveira
gradient_ratio_brahimi = high_freq_value_brahimi / low_freq_value_brahimi
print(f"Razão entre frequências baixa e alta: {gradient_ratio_oliveira}")
print(f"Razão entre frequências baixa e alta: {gradient_ratio_brahimi}")

# Cálculo da diferença média entre valores adjacentes
def mean_difference(matrix):
    difference = diff(matrix.flatten())
    return mean(difference)

gradiente_medio_oliveira = mean_difference(Q0)
gradiente_medio_brahimi = mean_difference(QB)
print(f"Gradiente médio entre valores adjacentes: {gradiente_medio_oliveira}")
print(f"Gradiente médio entre valores adjacentes: {gradiente_medio_brahimi}")



determinant_oliveira = linalg.det(Q0)
if isclose(determinant_oliveira, 0):
    print("A matriz é singular (não possui inversa).")
else:
    print(f"A matriz não é singular, determinante: {determinant_oliveira}")

determinant_brahimi = linalg.det(QB)
if isclose(determinant_brahimi, 0):
	print("A matriz é singular (não possui inversa).")
else:
	print(f"A matriz não é singular, determinante: {determinant_brahimi}")


"""
plt.plot(quality_factors, orto_oliveira, label='Tr Oliveira', marker='x', color='red', ls='None')
plt.plot(quality_factors, orto_brahimi, label='Tr Brahimi', marker='_', color='blue', ls='None')
plt.xlabel('Quality Factor')
plt.ylabel('Orthogonality')
plt.title('Analyses')
plt.legend()
plt.show()
"""
