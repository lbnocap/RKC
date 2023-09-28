import numpy as np   #改造二阶三步非线性例5
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import math
np.seterr(divide='ignore', invalid='ignore')
import cupy as cp



M=128
pi=np.pi
time_st=time.time()
x0=0
x_end=1
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
bt=0.03
af=0.1
A=np.zeros(((M-1)**2,(M-1)**2))
B=np.zeros((M-1,M-1)) 
E=np.eye(M-1)*af/(hx**2)
for i in range(0,M-1):
    if i==0:
       B[0][0],B[0][1]=(-4*af)/(hx**2),af/(hx**2)
    if 0<i<M-2:
       B[i][i-1],B[i][i],B[i][i+1]=af/(hx**2),(-4*af)/(hx**2),af/(hx**2)
    if i==M-2:
       B[i][i-1],B[i][i]=af/(hx**2),(-4*af)/(hx**2)
for j in range(0,M-1):
   if j==0:
      A[0:M-1,0:M-1],A[0:M-1,M-1:2*(M-1)]=B,E
   if 0<j<M-2:
      A[j*(M-1):(j+1)*(M-1),(j-1)*(M-1):(j)*(M-1)]=E
      A[j*(M-1):(j+1)*(M-1),(j)*(M-1):(j+1)*(M-1)]=B
      A[j*(M-1):(j+1)*(M-1),(j+1)*(M-1):(j+2)*(M-1)]=E
   elif j==M-2: 
      A[(j)*(M-1):(j+1)*(M-1),(j-1)*(M-1):(j)*(M-1)]=E
      A[j*(M-1):(j+1)*(M-1),(j)*(M-1):(j+1)*(M-1)]=B

eig1,abcd=np.linalg.eig(A)
eig2=np.max(np.abs(eig1)) 
print(eig2)