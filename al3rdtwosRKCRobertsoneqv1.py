import numpy as np   #Robertson equation
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import copy
import math

np.seterr(divide='ignore', invalid='ignore')
pi=math.pi
M=500
time_st=time.time()
x0=0
x_end=1
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
bt=1/(25* pi**2)
af=26/(25* pi**2)
gm=1/pi**2
e=np.zeros((M+1,1))
BB=np.zeros((3*M+3,3*M+3))
A=np.zeros((M+1,M+1)) 
B=np.zeros((M+1,M+1))
C=np.zeros((M+1,M+1))
y=np.zeros((3*M+3,1))
atol=1e-5
rtol=1e-5
for i in range(0,M+1):
    if i==0:
        y[i]=np.sin(pi*x[i])
        y[M+1+i]=0
        y[2*M+2+i]=1-np.sin(pi*x[i])
        e[0]=0
        B[0][0],B[0][1]=-2*bt/(hx**2),bt/(hx**2)
        B[0][M-1]=bt/(hx**2)
        A[0][0],A[0][1]=-2*af/(hx**2),af/(hx**2)
        A[0][M-1]=af/(hx**2)
        C[0][0],C[0][1]=-2*gm/(hx**2),gm/(hx**2)
        C[0][M-1]=gm/(hx**2)
    elif 0<i<M:
        y[i]=np.sin(pi*x[i])
        y[M+1+i]=0
        y[2*M+2+i]=1-np.sin(pi*x[i])
        e[i]=1
        B[i][i-1],B[i][i],B[i][i+1]=bt/(hx**2),-2*bt/(hx**2),bt/(hx**2)
        A[i][i-1],A[i][i],A[i][i+1]=af/(hx**2),-2*af/(hx**2),af/(hx**2)
        C[i][i-1],C[i][i],C[i][i+1]=gm/(hx**2),-2*gm/(hx**2),gm/(hx**2)
    else:
        y[i]=np.sin(pi*x[i])
        y[M+1+i]=0
        y[2*M+2+i]=1-np.sin(pi*x[i])
        e[i]=0 
        B[M][M],B[M][M-1]=-2*bt/(hx**2),bt/(hx**2)
        B[M][1]=bt/(hx**2)
        A[M][M],A[M][M-1]=-2*af/(hx**2),af/(hx**2)
        A[M][1]=af/(hx**2)
        C[M][M],C[M][M-1]=-2*gm/(hx**2),gm/(hx**2)
        C[M][1]=gm/(hx**2)

BB[0:M+1,0:M+1],BB[M+1:2*M+2,M+1:2*M+2]=A-0.04*np.eye(M+1),B
BB[2*M+2:3*M+3,2*M+2:3*M+3]=C
BB[M+1:2*M+2,0:M+1]=-0.04*np.eye(M+1)



def fun1(x,z):
    b=np.zeros((3*M+3,1))
    u=z[0:M+1]
    v=z[M+1:2*M+2]
    w=z[2*M+2:3*M+3]
    
    for j in range(0,M+1):
        b[j]=(10**4)*v[j]*w[j]
        b[M+1+j]=-3*(10**7)*(v[j]**2)-(10**4)*v[j]*w[j]
        b[2*M+2+j]=3*(10**7)*(v[j]**2)
    b=b.reshape((3*M+3,1))
    U=np.dot(BB,z).reshape((3*M+3,1))
    return U+b


def err(x,y,tc,h):
    x1=x.reshape((2*M+2,1))
    y1=y.reshape((2*M+2,1))
    z1=12*(x1-y1)
    return 0.1*(z1+6*h*(fun1(tc+h,x1)+fun1(tc+h,y1)))

def ro(x,y):
    e=1e-12;ln=len(y)
    Rv=y.copy()
    for j in range(ln):
        if y[j]==0:
            Rv[j]=e/2
        else:
            Rv[j]=y[j]*(1+e/2)
    e=max(e,e*np.linalg.norm(Rv,ord=2))
    Rv1=y.copy()
    f1=fun1(x,Rv1) 
    f2=fun1(x,Rv)
    Rv1=Rv+e*(f1-f2)/(np.linalg.norm(f1-f2))
    Rv1=Rv1.reshape((ln,1))
    f1=fun1(x,Rv1)
    R=np.linalg.norm(f1-f2)/e
    Rr=R
    fg=R;fg1=0
    while fg > 1e-4*R and fg1<20:
        Rv1=Rv+e*(f1-f2)/np.linalg.norm(f1-f2)
        f1=fun1(x,Rv1)
        R=np.linalg.norm(f1-f2)/e
        fg=np.abs(R-Rr)
        fg1+=1
        Rr=R 
    if fg1==20:
        R=1.2*R
    return R,fg1

widetwoRKCv2= np.load('widetwostepRKCv2.npz', allow_pickle=True)
cs = widetwoRKCv2['cs']
us1=widetwoRKCv2['us1']
vs1=widetwoRKCv2['vs1']
vs=widetwoRKCv2['vs']
us=widetwoRKCv2['us']
bs=widetwoRKCv2['bs']
xxs=widetwoRKCv2['xxs']


