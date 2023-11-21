import cv2
import csv
import numpy as np

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 / 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(int(self.frame_len))
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print (str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return ret, bgr


if __name__ == "__main__":
    path = '/media/tltsilveira/FAST-DATA/CTC360/8 bits/'
    with open('videos360.csv', 'r') as csvfile:
       spamreader = csv.reader(csvfile, quotechar='"', delimiter=',')
       for row in spamreader:
          filename, h, w = row
          size = (int(h),int(w))
          filenamef = path+filename+'/'+filename+'.yuv'
          print(filenamef, size); print('')
    
          cap = VideoCaptureYUV(filenamef, size)
          tamanho=0
          while 1:
              ret, frame = cap.read()
              if ret:
                  cv2.waitKey(30)
                  tamanho+=1
              else:
                  break
          #print(tamanho)
          idx = 0
          cap = VideoCaptureYUV(filenamef, size)
          while 1:
              ret, frame = cap.read()
              if ret:
                  if idx == 0 or idx == tamanho//2 or idx == tamanho-1:
                      cv2.imwrite(filename+'_'+str(idx)+'.bmp', frame)
                      #cv2.imshow("frame", frame)
                      #cv2.waitKey(30) 
                  idx = idx + 1
              else:
                  break
          
