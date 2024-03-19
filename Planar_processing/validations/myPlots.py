import os
from scipy import signal
from pdb import set_trace as pause
from matplotlib import pyplot as plot
from operator import itemgetter
import csv

dados = []
methods = 0

with open('standards.csv', 'r') as file:
    reader = csv.DictReader(file)
    for linha in reader:
        dados.append({'imageName':linha['File name'], 'method': linha['Method'], 'psnr': linha['PSNR'], 'ssim': linha['SSIM'], 'bpp':linha['BPP']})
        

buffer_methods = []
for dado in dados:
    buffer_methods.append(dado['method'])
methods = len(set(buffer_methods))

print(f'Quantidade de metodos: {methods}')