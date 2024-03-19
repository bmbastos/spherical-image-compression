import matplotlib.pyplot as plt
from pdb import set_trace as pause

import sys
from numpy import *
from time import time	
from glob import glob
from skimage.io import imread

from tools import *

def QMatrix(QF):

   #QF = 50 # A quantizacao nao deve funcionar nos casos extremos (QF = 0 e QF = 100)
  
   def __S(QF):
      return 5000./QF if QF < 50 else 200 - 2.*QF

   Q0 = asarray([[16, 11, 10, 16, 24, 40, 51, 61], 
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],  # ITU and Simone et al
                  #   [14, 17, 22, 29, 51, 84, 80, 62],  # Bayer et al
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

   #assert all(isclose(floor((__S(50)*Q0+50)/100.), Q0))
   return floor((__S(QF)*Q0+50)/100.)
  

class DeSimoneQuantMatrix:

    kLUT = None
    minLUT = None
    maxLUT = None

    Q = None

    def __init__(self, imgHeight, Q = QMatrix):
        self.__buildLUT(imgHeight)
        self.Q = Q

    def __str__(self):
        return "Implementation of 10.1109/PCS.2016.7906402"

    def __mapKandEl(self, rowIndex, imgHeight):
        el = rowIndex/imgHeight * pi - pi/2.
        kprime = arange(8)
        k = clip(0, 7, around(kprime/cos(el))).astype(int)
        return (k, el)

    def __buildLUT(self, imgHeight):
        N = 8
        ks, els = [], []
        rowIdxs = arange(0, imgHeight+1, N)

        for rowIdx in rowIdxs:
            k, el = self.__mapKandEl(rowIdx, imgHeight)
            ks += [k]; els += [el]
        ks = asarray(ks); els = abs(asarray(els))

        self.kLUT = unique(ks, axis=0)
        self.minLUT = asarray([finfo('f').max for x in self.kLUT])
        self.maxLUT = []
        auxMaxLUT = [finfo('f').min for x in self.kLUT]

        for idx in range(len(ks)):
            for idx2 in range(len(self.kLUT)):
                if sum(ks[idx] - self.kLUT[idx2]) == 0:
                    if els[idx] > auxMaxLUT[idx2]: 
                        auxMaxLUT[idx2] = els[idx]
                    if els[idx] < self.minLUT[idx2]: 
                        self.minLUT[idx2] = els[idx] 
     
        for idx in range(len(self.kLUT)):
            if idx != len(self.kLUT)-1: 
                self.maxLUT += [self.minLUT[idx+1]]
            else: 
                self.maxLUT += [auxMaxLUT[idx]]
     
        self.maxLUT = asarray(self.maxLUT)

    def printLUT(self):
        for idx in range(len(self.kLUT)):
            print(self.kLUT[idx], "%.4f" % self.minLUT[idx], "%.4f" % self.maxLUT[idx])
        

    def QtildeAtEl(self, el, QF = 50):
        ks = None
        el = abs(el) # LUT is mirrored
        for idx in range(len(self.kLUT)):
            if el >= self.minLUT[idx] and el < self.maxLUT[idx]: 
                ks = self.kLUT[idx]
        if ks is None and el == 0: ks = self.kLUT[0]
        Q = vstack(([self.Q(QF)[:,k] for k in ks])).T
        return Q
    








def DCT(N):
   C = zeros((N,N))
   for i in range(N):
      for j in range(N):
         alpha = 1./sqrt(N) if i == 0 else sqrt(2./N)
         C[i,j] = alpha*cos(pi*i*(2*j+1)/(2*N))
   return C

def encodeQuantiseNDecode(I, T, Q, N = 8):
    h, w = I.shape
    A = Tools.umount(I, (N, N))# - 128
    Aprime1 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T, A), T.T) # forward transform
    Aprime2 = multiply(Q, around(divide(Aprime1, Q))) # quantization
    Aprime3 = einsum('mij, jk -> mik', einsum('ij, mjk -> mik', T.T, Aprime2), T) # inverse transform
    B = Tools.remount(Aprime3, (h, w)) #+ 128
    return Aprime2.reshape(h,w), B 

def prepareQPhi(QF, h, w, N = 8):
    qHandler = DeSimoneQuantMatrix(h) 
    els = linspace(-pi/2, pi/2, h//N+1)
    els = 0.5*(els[1:] + els[:-1]) # gets the "central" block elevation
    QPhi = asarray([qHandler.QtildeAtEl(el = el, QF = QF) for el in els])
    QPhi = repeat(QPhi, w//N, axis=0)
    #plt.imshow(Tools.remount(QPhi, (h, w))); plt.show() # plot the quantization matrices map
    return QPhi





#'''
if __name__ == "__main__":
    

    #FName = 'PoleVault_le_3840x1920_30fps_8bit_420_erp_0.bmp'
    FName = './test_images/sample-ERP.jpg'

    I = around(255*imread(FName, True))
    #plt.imshow(I, cmap='gray'); plt.show() # original


    QF = 50
    C = DCT(8)
    QPhi = prepareQPhi(QF, I.shape[0], I.shape[1])
    _, Ir = encodeQuantiseNDecode(I, C, QPhi) # first value helps computing bpp
    Ir = clip(Ir, 0, 255)

    plt.imshow(Ir, cmap='gray'); plt.show() # decompressed image
    plt.imshow(Ir[180: 188, 360:368], cmap='gray'); plt.show()

    #print(I.shape, Ir.shape)
    #plt.imshow(abs(I-Ir)); plt.show() #error 
        
#'''   
