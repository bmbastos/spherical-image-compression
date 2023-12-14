import matplotlib.pyplot as plt
from pdb import set_trace as pause

import sys
from numpy import *
from time import time	
from glob import glob
from skimage.io import imread
from matplotlib import pyplot as plot

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
        Q = vstack(([self.Q(QF)[:,k] for k in ks])).T
        return Q
    



#'''
if __name__ == "__main__":
    
    QF = 50
    #Q = QMatrix(QF)

    height = 1920    

    qHandler = DeSimoneQuantMatrix(height) 
    print(qHandler)

    qHandler.printLUT()
    Q = qHandler.QtildeAtEl(el = pi/4., QF = QF)
    print(Q)

    ## Continuação Bruno Bastos

    image = imread("./omnidirecional_images/DrivingInCity_3840x1920_30fps_8bit_420_erp_0.bmp")
    plot.imshow(image)
    plot.show()

#'''   
