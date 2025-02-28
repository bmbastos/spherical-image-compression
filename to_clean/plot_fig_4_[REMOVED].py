from skimage.io import imread, imsave
from numpy import *

from skimage.draw import rectangle_perimeter

import glob
import sys
from scipy.ndimage import zoom
from pdb import set_trace as pause
import sys
from matplotlib import pyplot as plt

filenames = glob.glob('*.png')

for filename in filenames:
   print(filename)
   #filename = sys.argv[-1]
   img = imread(filename)[:,:,0]
   #print(img.min(),img.max())

   img = dstack((img,img,img,255*ones_like(img)))

   f = 3.75
   y1,y2,x1,x2 = 110, 260, 350, 450 
   #ßy1,y2,x1,x2 = 40, 50, 35, 65 
   y1,y2,x1,x2 = int(f*y1), int(f*y2), int(f*x1), int(f*x2)
   crop = img[x1:x2,y1:y2].copy()
   #print(crop.shape)
   crop = zoom(crop,(2.5,2.5,1),mode='nearest')

   #plt.imshow(crop);plt.show();sys.exit()
   #print(img.shape, crop.shape)

   ys=50
   xs=50
   ys = int(f*ys)   
   xs = int(f*xs)
   h, w, _ = img.shape
   hc, wc, _ = crop.shape
   img[ys:hc+ys:,w-wc-xs:-xs] = crop

   thickness = 3
   for i in range(0,thickness):
      start = (x1-i, y1-i)
      end = (x2+i, y2+i)
      rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
      img[rr,cc] = (255,0,0,255)

      start = (ys-i, w-wc-xs-i)
      end = (hc+ys-1+i, w-xs-1+i)
      rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
      img[rr,cc] = (0,0,255,255)

   #pause()

   imsave(filename.replace('.png', '_roi2.png'), img)

