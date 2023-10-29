import numpy as np   #原二阶二步
import numpy.matlib
import matplotlib.pyplot as plt
import math
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
from numpy.polynomial import chebyshev
import time
np.seterr(divide='ignore', invalid='ignore')
M=15000
time_st=time.time()
x0=-math.pi
x_end=math.pi
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
A=np.zeros((M+1,M+1))
y=np.zeros((M+1,1))
solu=np.zeros((M+1,1))
bt=0.1
af=0
A=np.zeros((M+1,M+1))
tol=1e-4        
A[0][0],A[0][1],A[0][2],A[0][3]=2*bt/(hx**2)+af/hx,-5*bt/(hx**2)-af/hx,4*bt/(hx**2),-bt/(hx**2)
A[1][0],A[1][1],A[1][2]=bt/(hx**2)+af/hx,-2*bt/(hx**2)-af/hx,bt/(hx**2)
for i in range(M+1):
    y[i]=math.sin(x[i])
    solu[i]=np.e**(-bt*2)*np.sin(x[i]-af*2)
    if i>=2 and i<=M-1:
        A[i][i-2],A[i][i-1]=-af/(2*hx),bt/(hx**2)+4*af/(2*hx)
        A[i][i],A[i][i+1]=-2*bt/(hx**2)-3*af/(2*hx),bt/(hx**2)
    '''
    if i==M:
        A[M][M-3],A[M][M-2]=-bt/(hx**2),4*bt/(hx**2)-af/hx
        A[M][M-1],A[M][M]=-5*bt/(hx**2)+4*af/hx,2*bt/(hx**2)-3*af/hx
    '''
    if i==M:
        A[M][1],A[M][M-2]=bt/(hx**2),-af/(2*hx)
        A[M][M-1],A[M][M]=bt/(hx**2)+4*af/(2*hx),-2*bt/(hx**2)-3*af/hx

#eig1,abcd=np.linalg.eig(A)
#eig2=np.max(np.abs(eig1))
print(A)
