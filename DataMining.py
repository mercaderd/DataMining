__author__ = 'Daniel'


import sys, getopt
from glob import glob
from os import path
import numpy as np
import imagetools
import pywt
from sklearn import svm



def LoadDataFromFolder(folder = '.', ftype = '*.dat'):
   filelist = glob(path.join(folder,ftype))
   firstpattern = np.loadtxt(filelist[1],dtype=np.float32)
   firstpattern = firstpattern[:,2]
   n = len(filelist)
   d = firstpattern.size
   X = np.zeros([n,d])
   y = np.zeros([n,1], dtype = np.int32)
   for i in range(len(filelist)):
      data = np.loadtxt(filelist[i],dtype=np.float32)
      data = imagetools.arraytoimage(data[:,2])
      X[i,:] = data
      if "BKGND" in filelist[i]: y[i] = 0
      if "ECH" in filelist[i]: y[i] = 1
      if "NBI" in filelist[i]: y[i] = 2
      if "STRAY" in filelist[i]: y[i] = 3
   return X,y


def ReduceDimension(X = np.zeros([2,2])):
    r, c = X.shape
    image = X[0,:].reshape([385,576])
    coeffs = pywt.wavedec2(image,'db1', level=4)
    cA4, (cH4, cV4, cD4), (cH3, cV3, cD3),(cH2, cV2, cD2),(cH1, cV1, cD1) = coeffs
    nr,nc = cA4.shape
    rX = np.zeros([r,nc*nr], dtype=np.float32)
    for i in range(r):
        image = X[i,:].reshape([385,576])
        coeffs = pywt.wavedec2(image,'db1', level=4)
        cA4, (cH4, cV4, cD4), (cH3, cV3, cD3),(cH2, cV2, cD2),(cH1, cV1, cD1) = coeffs
        rX[i,:] = cV4.flatten()
    return rX

def TrainSVM(X,y):
    clf = svm.SVC()
    clf.fit(X, y.ravel())
    print(clf)
    print(y.ravel())
    return clf

def main():
    X,y = LoadDataFromFolder('C:\\Users\\Daniel\\Google Drive\\Master ISC\\Tercer Curso\\1C Mineria de Datos\\Datos\\SenalesTJII\\Imagenes')
    rX = ReduceDimension(X)
    C=TrainSVM(rX,y)
    print(C)

if __name__ == "__main__":
   main()
