from numpy import *
from pdb import set_trace as pause
from matplotlib import pyplot as plt

def S(QF): 
   return 5000/QF if QF < 50 else 200-2*QF
   
def np2_round(x):
   return 2**round(log2(x))
   
def np2_floor(x):
   return 2**floor(log2(x))

def np2_ceil(x):
   return 2**ceil(log2(x))

QFs = arange(1,99+1) # problem with QP = 100
Ss = asarray([S(qf) for qf in QFs])

Np2r = asarray([np2_round(s) for s in Ss])
Np2f = asarray([np2_floor(s) for s in Ss])
Np2c = asarray([np2_ceil(s) for s in Ss])

plt.plot(QFs, Ss, label='S')
plt.plot(QFs, Np2r, label='np2_round')
plt.plot(QFs, Np2f, label='np2_floor')
plt.plot(QFs, Np2c, label='np2_ceil')
plt.legend()
plt.show()

print(len(unique(Ss)), len(unique(Np2r)), len(unique(Np2f)), len(unique(Np2c)))

pause()


