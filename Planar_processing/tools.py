from math import *
from numpy import *
import pprint

class Tools(object):
  
  @staticmethod
  def selectCoefficients(data, r):
    if len(data.shape) == 1 or data.shape[0] == 1: # one dimensional
      dimension = data.shape[1]
      index = 0
      selected = []
      for value in data.flat:
        if index < r:
          selected += [value]
        else:
          break
        index += 1
        selected += zeros(dimension - r).tolist()
        return(matrix(selected).reshape((1,dimension)))
    else: # is two dimensional (tested for power of two sizes)
      order = data.shape[0]
      selected = zeros(order**2).reshape((order,order))
      i = j = index = 0
      increasing = True
      while index < r:
        selected[i,j] = data[i,j]
        if (i == 0 or i == order-1) and j % 2 == 0:
          j += 1
        elif ((i == 0) and j % 2 == 1) or (j == order - 1 and i % 2 == 0 and i > 0):
          i += 1
          j -= 1
          increasing = False	
        elif (i == order - 1 and j % 2 == 1) or (j == 0 and i % 2 == 0):
          i -= 1
          j += 1
          increasing = True
        elif (j == 0 or j == order - 1) and i % 2 == 1:
          i += 1 
        else:
          if increasing:
            j += 1
            i -= 1
          else:
            j -= 1
            i += 1
        index += 1
      return(selected)
    
  @staticmethod
  def __chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
    
  @staticmethod
  def umount(data, newdimension):
    if len(newdimension) == 1:
      return(list(Tools.__chunks(data, newdimension[0])))
    elif len(newdimension) == 2:
      nrows, ncols = newdimension
      h, w = data.shape
      return (data.reshape(h//nrows, nrows, -1, ncols)
		.swapaxes(1,2)
		.reshape(-1, nrows, ncols))
  
  @staticmethod
  def remount(data, newdimension):
    if len(newdimension) == 1:
      returnedList = []
      for chunk in data:
	      for value in chunk:
	        returnedList += [value]
      return(returnedList)
    elif len(newdimension) == 2:
      h, w = newdimension
      n, nrows, ncols = data.shape
      return (data.reshape(h//nrows, -1, nrows, ncols)
		.swapaxes(1,2)
		.reshape(h, w))
		
		
		
