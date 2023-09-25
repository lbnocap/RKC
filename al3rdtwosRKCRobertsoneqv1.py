import numpy as np   #改造二步三阶定步长 Robertson equation
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import copy
import math
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')
pi=math.pi 
M=400
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
solu=np.zeros((3*M+3,1))
y=np.zeros((3*M+3,1))
atol=1e-5
rtol=1e-5
for i in range(0,M+1):
    if i==0:
        y[i]=np.sin(pi*x[i])
        y[M+1+i]=0
        y[2*M+2+i]=1-np.sin(pi*x[i])
        solu[i]=np.exp(-1)*np.sin(pi*x[i])
        solu[M+1+i]=0
        solu[2*M+2+i]=1-np.exp(-1)*np.sin(pi*x[i])
        e[0]=0
        B[0][0],B[0][1]=-2*bt/(hx**2),bt/(hx**2)
        B[0][M-1]=-bt/(hx**2)
        A[0][0],A[0][1]=-2*af/(hx**2),af/(hx**2)
        A[0][M-1]=-af/(hx**2)
        C[0][0],C[0][1]=2*gm/(hx**2),-5*gm/(hx**2)
        C[0][2],C[0][3]=4*gm/(hx**2),-gm/(hx**2)
    elif 0<i<M:
        y[i]=np.sin(pi*x[i])
        y[M+1+i]=0
        y[2*M+2+i]=1-np.sin(pi*x[i])
        solu[i]=np.exp(-1)*np.sin(pi*x[i])
        solu[M+1+i]=0
        solu[2*M+2+i]=1-np.exp(-1)*np.sin(pi*x[i])
        e[i]=1
        B[i][i-1],B[i][i],B[i][i+1]=bt/(hx**2),-2*bt/(hx**2),bt/(hx**2)
        A[i][i-1],A[i][i],A[i][i+1]=af/(hx**2),-2*af/(hx**2),af/(hx**2)
        C[i][i-1],C[i][i],C[i][i+1]=gm/(hx**2),-2*gm/(hx**2),gm/(hx**2)
    else:
        y[i]=np.sin(pi*x[i])
        y[M+1+i]=0
        y[2*M+2+i]=1-np.sin(pi*x[i])
        solu[i]=np.exp(-1)*np.sin(pi*x[i])
        solu[M+1+i]=0
        solu[2*M+2+i]=1-np.exp(-1)*np.sin(pi*x[i])
        e[i]=0 
        B[M][M],B[M][M-1]=-2*bt/(hx**2),bt/(hx**2)
        B[M][1]=-bt/(hx**2)
        A[M][M],A[M][M-1]=-2*af/(hx**2),af/(hx**2)
        A[M][1]=-af/(hx**2)
        C[M][M],C[M][M-1]=2*gm/(hx**2),-5*gm/(hx**2)
        C[M][M-2],C[M][M-3]=4*gm/(hx**2),-gm/(hx**2)

BB[0:M+1,0:M+1],BB[M+1:2*M+2,M+1:2*M+2]=A,B
BB[2*M+2:3*M+3,2*M+2:3*M+3]=C


def fun1(x,z):
    b=np.zeros((3*M+3,1))
    u=z[0:M+1]
    v=z[M+1:2*M+2]
    w=z[2*M+2:3*M+3]
    
    for j in range(0,M+1):
        b[j]=(10**4)*v[j]*w[j]-0.04*u[j]
        b[M+1+j]=-3*(10**7)*(v[j]**2)-(10**4)*v[j]*w[j]+0.04*u[j]
        b[2*M+2+j]=3*(10**7)*(v[j]**2)
    b=b.reshape((3*M+3,1))
    U=np.dot(BB,z).reshape((3*M+3,1))
    return U+b



def err(x,y,tc,h):
    x1=x.reshape((3*M+3,1))
    y1=y.reshape((3*M+3,1))
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
    while fg > 1e-4*R and fg1<30:
        Rv1=Rv+e*(f1-f2)/np.linalg.norm(f1-f2)
        f1=fun1(x,Rv1)
        R=np.linalg.norm(f1-f2)/e
        fg=np.abs(R-Rr)
        fg1+=1
        Rr=R 
    if fg1==30:
        R=1.1*R
    return R,fg1

widetwoRKCv2= np.load('widetwostepRKCv2.npz', allow_pickle=True)
cs = widetwoRKCv2['cs']
us1=widetwoRKCv2['us1']
vs1=widetwoRKCv2['vs1']
vs=widetwoRKCv2['vs']
us=widetwoRKCv2['us']
bs=widetwoRKCv2['bs']
xxs=widetwoRKCv2['xxs']


def RKC(fun1,t0,t_end,h,u0,s): 
    
    tc=[t0] #t的初始
    y=u0
    counter=0
    fg1=0
    nfe=0
    s_max=0
    h1=h
    lop=0
    yb0=np.zeros((2*M+2,1))
    while tc[-1]<t_end:
        c=cs[s,0]
        u1=us1[s,0]
        u=us[s,0]
        v1=vs1[s,0]
        v=vs[s,0]
        xx=xxs[s,0]
        nfe=s+nfe+fg1+3
        k0=np.zeros((3*M+3,1))
        k1=np.zeros((3*M+3,1))
        k2=np.zeros((3*M+3,1))
        k3=np.zeros((3*M+3,1))
        ky0=np.zeros((3*M+3,1))
        ky1=np.zeros((3*M+3,1))
        k0=y[:,-1].copy()
        k0=k0.reshape((3*M+3,1))
        ky0=fun1(tc[-1],k0)
        k1=k0+u1[1] *h *ky0
        ky1=fun1(tc[-1]+u1[1]*h,k1)
        k2=k1.copy()
        k1=k0.copy()
        if tc[-1]==0:
            print(1)
        for j in range(2,s+1):
            k3=u[j]*k2+v[j]*k1+(1-u[j]-v[j])*k0+u1[j]*h*ky1+v1[j]*h*ky0
            #if j==8:
              #  print(k3)
            ky1=fun1(tc[-1]+c[j]*h,k3)
            k1=k2.copy()
            k2=k3.copy()
        xx1=xx[0]
        xx2=xx[1]
        xx3=xx[2]
        xx4=xx[3]
            
        if counter==0:
            yc=k3.copy()
           
           # print("yc:",yc)
          

            #err2=err(y[:,-1],yc,h1)
            #err1=np.linalg.norm(err2)/math.sqrt(3*M+3)
            yb0=k3.copy()
           # print(yb0)y, yc))
            y = np.column_stack((y, yc))
            counter+=1
            tc.append(tc[-1]+h1)
            pu,fg1=ro(tc[-1]+h1,yc)
            s2=math.sqrt(h1*pu/0.4)
            s=math.ceil(s2)
            if s_max<s:
                   s_max=s
            if s>250:
                s=250

            if s<5:
                s=5  
        else :
            k02=y[:,-2].copy()
            k02=k02.reshape((3*M+3,1))
            yb=k3.copy()    
            yc=xx1*k02+xx2*yb0+xx3*k0+xx4*yb
            yb0=yb.copy()
            if tc[-1]==0.001:
               1# print(yc,yb)
            tc.append(tc[-1]+h1)
            pu,fg1=ro(tc[-1]+h1,yc)
            if pu>lop:
                lop=pu
            if tc[-1] + h1 > t_end:
                 h1 = t_end -tc[-1]
                 h=h1  
            s2=np.sqrt(h1*pu/0.4)                                           
            s=math.ceil(s2)
            if s<6:
                s=5
            if s>s_max:
                s_max=s 
            if s>250:
                s=250
            #h=h1
            #err2=err(y[:,-1],yc,h1)
            #err1=np.linalg.norm(err2)/math.sqrt(3*M+3)
            #print(err1)
            y = np.column_stack((y, yc))
    return np.array(tc),np.array(y),nfe,s_max,lop


t0=0
t_end=1
h=0.0005
eig3,fg1=ro(0,y)
s2=np.sqrt(h*eig3/0.4)                                           
s=math.ceil(s2)
print(s)
print('eig:',eig3)
#print(fun1(x,y))
#print(y)
if s<=5:
    s=5
#tc1,y1,nfe1,s_max1=RKC2(fun1,t0,t_end,0.0001,y,s)
tc,y,nfe,s_max,lop=RKC(fun1,t0,t_end,h,y,s)
#mse = np.mean((np.array(y[1:M,-1]) - np.array(solu[1:M]))**2)
#mae = np.mean(np.abs(np.array(y[1:M,-1]) - np.array(solu[1:M])
# ))
#err=sum([(x - y) ** 2 for x, y in zip(y[:,-2], y1[:,-2])] )/ len(y1[:,-2])
#print("terrpr:",np.sqrt(err))
time_end=time.time()
print(time_end-time_st)
print(tc)
print("步数：",len(tc))
print("评估次数：",nfe)
print("s_max:",s_max)
print("lop:",lop)
err2=err(y[:,-3],y[:,-2],0,h)
#print(y[:,3])
err1=np.linalg.norm(err2)/math.sqrt(3*M+3)
print("err1:",err1)
solu1=np.load('Robertsoneqsolu.npy')
err=sum([(x - y) ** 2 for x, y in zip(y[1:3*M+2,-1], solu1[1:3*M+2])] )/ len(solu1[1:3*M+2])
print("err:",np.sqrt(err))
