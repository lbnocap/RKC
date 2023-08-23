import numpy as np   #改造二阶二步非线性例5
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import math
np.seterr(divide='ignore', invalid='ignore')

M=500
pi=np.pi
time_st=time.time()
x0=0
x_end=1
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
bt=0.03
gm=-1
e=np.zeros((M+1,1))
A=np.zeros((2*M+2,M*2+2))
B=np.zeros((M+1,M+1)) 
y=np.zeros((2*M+2,1))
solu=np.zeros((2*M+2,1))
tol=1e-3 
for i in range(0,M+1):
    if i==0:
        solu[i]=np.exp(-((pi)**2) *bt*1)*np.sin(pi*(x[i]-1))
        solu[M+1+i]=np.exp(-((pi)**2) *bt*1)*np.cos(pi*(x[i]-1))
        y[i]=np.sin(pi*x[i])
        y[M+1+i]=np.cos(pi*x[i])
        B[0][0],B[0][1]=-2*bt/(hx**2)+gm,bt/(hx**2)
        B[0][M-1]=-bt/(hx**2)
        B[M][M-1],B[M][M]=bt/(hx**2),-2*bt/(hx**2)+gm
        B[M][1]=-bt/(hx**2)
    elif 0<i<M:
        solu[i]=np.exp(-((pi)**2) *bt*1)*np.sin(pi*(x[i]-1))
        solu[M+1+i]=np.exp(-((pi)**2) *bt*1)*np.cos(pi*(x[i]-1))
        y[i]=np.sin(pi*x[i])
        y[M+1+i]=np.cos(pi*x[i])
        B[i][i-1],B[i][i],B[i][i+1]=bt/(hx**2),-2*bt/(hx**2)+gm,bt/(hx**2)
    elif i==M:
        solu[i]=np.exp(-((pi)**2) *bt*1)*np.sin(pi*(x[i]-1))
        solu[M+1+i]=np.exp(-((pi)**2) *bt*1)*np.cos(pi*(x[i]-1))
        y[i]=np.sin(pi*x[i])
        y[M+1+i]=np.cos(pi*x[i])

A[0:M+1,0:M+1]=B
A[M+1:2*M+2,M+1:M*2+2]=B
eig1,abcd=np.linalg.eig(A)
eig2=np.max(np.abs(eig1))
def g1(t,x):
    return np.exp(-((pi)**2) *bt*t)*(-gm*np.sin(pi*(x-t))-pi*np.cos(pi*(x-t)))+np.exp(-3*((pi)**2) *bt*t)*(np.sin(pi*(x-t))**2)*np.cos(pi*(x-t))
def g2(t,x):
    return np.exp(-((pi)**2) *bt*t)*(-gm*np.cos(pi*(x-t))-pi*np.sin(pi*(x-t)))+np.exp(-3*((pi)**2 )*bt*t)*(np.cos(pi*(x-t))**2)*np.sin(pi*(x-t))
def fun1(t,z):
    U=np.dot(A,z).reshape((2*M+2,1))
    b=np.zeros((2*M+2,1))
    u=z[0:M+1]
    v=z[M+1:2*M+2]
    for j in range(0,M+1):
        b[j]=-(u[j]**2)*v[j]+g1(t,x[j])
        b[M+1+j]=-(v[j]**2)*u[j]+g2(t,x[j])
    b=b.reshape((1002,1))
    return U+b
def RKC(f,t0,t_end,h,u0,s):
    h_v=[h]
    h1=h
    tc=[t0] #t的初始
    y=u0
    cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
    cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
    t3=cheb_poly.deriv(1)
    t4=cheb_poly.deriv(2)
    t5=cheb_poly.deriv(3)
    t42=chebyshev.Chebyshev([0] * (2 + 1))
    t42.coef[-1]=1
    t22=t42.deriv(2)
    counter=0
    while tc[-1]<t_end: 
        w0=1+(2/13)/((s)**2)
        c=np.zeros(s+1)
        b=np.zeros(s+1)
        t=np.zeros(s+1)
        t1=np.zeros(s+1)
        k=np.zeros((2*M+2,4))
        k0=np.zeros((2*M+2,1))
        k1=np.zeros((2*M+2,1))
        k2=np.zeros((2*M+2,1))
        k3=np.zeros((2*M+2,1))
        ky=np.zeros((2*M+2,2))
        ky0=np.zeros((2*M+2,1))
        ky1=np.zeros((2*M+2,1))
        c[0]=0
        b[0]=1
        t[0]=1
        t[1]=w0
        t1[0]=0
        t1[1]=1
        u=np.zeros(s+1)
        u[0],u[1]=0,0
        v=np.zeros(s+1)
        v1=np.zeros(s+1)
        v[0],v[1]=0,0  
        v1[0],v1[1]=0,0
        u1=np.zeros(s+1)
        for j in range(2,s+1):
         t[j]=2*w0*t[j-1]-t[j-2]
         t1[j]=2*t[j-1]+2*w0*t1[j-1]-t1[j-2]
        b[0]=b[1]=b[2]=t22(w0)/(t1[2]**2)
        w1=t3(w0)/t4(w0) 
        u[0],u1[1]=0,b[1]*w1
        k0=y[:,-1]
        k0=k0.reshape((2*M+2,1))
        ky0=fun1(t[-1],k0)
        k1=k0+u1[1] *h *ky0
        ky1=fun1(t[-1]+u1[1]*h,k1)
        c[1]=u1[1]
        k2=k1
        k1=k0
        for j in range(2,s+1):
            cheb_poly1 = chebyshev.Chebyshev([0] * (j + 1))
            cheb_poly1.coef[-1] = 1
            tj=cheb_poly1.deriv(2)
            b[j]=tj(w0)/(t1[j]**2)
            u[j]=2*w0*b[j]/b[j-1]
            #print(u[j])
            v[j]=-b[j]/b[j-2]
            #print(v[j])
            u1[j]=2*w1*b[j]/b[j-1]
            v1[j]=-(1-b[j-1]*t[j-1])*u1[j]
            c[j]=u[j]*c[j-1]+v[j]*c[j-2]+u1[j]+v1[j]
            k3=u[j]*k2+v[j]*k1+(1-u[j]-v[j])*k0+u1[j]*h*ky1+v1[j]*h*ky0
            #if j==4:
                #print(k[4])
            ky1=fun1(tc[-1]+c[j]*h,k3)
            k1=k2
            k2=k3
        r=1
        cc=t3(w0)*t5(w0)/(t4(w0)**2)
        yt=1/np.sqrt(cc)
        #yt=0.6
        bn=(1+r)/(yt*(1+r*yt))
        bf1=(r**2)*(1-yt)/(1+yt*r)
        b0=1-bn-bf1
        h_v.append(h1)
        tc.append(tc[-1]+h1)
        if counter==0:
            yc=k3
            counter+=1
            #h=yt*h
        else :
            yc=k3
           #yc=bf1*y[:,-2]+b0*y[:,-1]+bn*k[:,3]
        y = np.column_stack((y, yc))
    return np.array(tc),np.array(y)
t0=0
t_end=1
h=0.01
eig1,abcd=np.linalg.eig(A)
eig2=np.max(np.abs(eig1))
print(eig2)
s2=math.sqrt(h*eig2/0.45)
s=math.ceil(s2)
print(s)
if s<3:
    s=2
tc,y=RKC(fun1,t0,t_end,h,y,s)
err=sum([(x - y) ** 2 for x, y in zip(y[1:M,-1], solu[1:M])] )/ len(solu[1:M])
print(np.sqrt(err))
print(y[:,-1])

