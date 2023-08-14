import numpy as np   #原二阶二步
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
np.seterr(divide='ignore', invalid='ignore')
M=200
x0=0
x_end=1
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
bt=0.04
bf=0.001
BB=np.zeros((3*M+3,3*M+3))
B=np.zeros((M+1,M+1))
tol=1e-3
B[0][0],B[0][1]=-2*bt/(hx**2),bt/(hx**2)
B[0][M-1]=bt/(hx**2)
B[M][M],B[M][M-1]=-2*bt/(hx**2),bt/(hx**2)
B[M][1]=bt/(hx**2)
for i in range(1,M):
    B[i][i-1],B[i][i],B[i][i+1]=bt/(hx**2),-2*bt/(hx**2),bt/(hx**2)
BB[0:M+1,0:M+1],BB[M+1:2*M+2,M+1:2*M+2]=B,B
BB[2*M+2:3*M+3,2*M+2:3*M+3]=B+np.eye(M+1)
BB[0:M+1,2*M+2:3*M+3]=(-1/bf)*np.eye(M+1)
BB[M+1:2*M+2,2*M+2:3*M+3]=np.eye(M+1)
BB[2*M+2:3*M+3,0:M+1],BB[2*M+2:3*M+3,M+1:2*M+2]=-0.4*np.eye(M+1),-1*np.eye(M+1)
A=BB
eig1,abcd=np.linalg.eig(A)
eig2=np.max(np.abs(eig1))
print(eig2)
print(BB)

