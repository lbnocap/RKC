import numpy as np   #rkc2定步长 Robertson equation
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import copy
import math
import pandas as pd



np.seterr(divide='ignore', invalid='ignore')
M=200
pi=math.pi
time_st=time.time()
x0=0
x_end=1
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
bt=0.04
bf=0.001
e=np.zeros((M,1))
BB=np.zeros((3*M,3*M))      
B=np.zeros((M,M)) 
y=np.zeros((3*M,1))
tol=1e-3

for i in range(0,M):
    if i==0:   
        y[i]=0
        y[M+i]=-2*np.cos(2*np.pi*x[i+1])
        y[2*M+i]=2*np.sin(2*np.pi*x[i+1])
        e[0]=0
        B[0][0],B[0][1]=-2*bt/(hx**2),bt/(hx**2)
        B[0][M-1]=bt/(hx**2)
   
    elif 0<i<M-1:
        y[i]=0
        y[M+i]=-2*np.cos(2*np.pi*x[i+1])
        y[2*M+i]=2*np.sin(2*np.pi*x[i+1])
        e[i]=1
        B[i][i-1],B[i][i],B[i][i+1]=bt/(hx**2),-2*bt/(hx**2),bt/(hx**2)
    else:
        y[i]=0
        y[M+i]=-2*np.cos(2*np.pi*x[i+1])
        y[2*M+i]=2*np.sin(2*np.pi*x[i+1])
        e[i]=0 
        B[M-1][M-1],B[M-1][M-2]=-2*bt/(hx**2),bt/(hx**2)
        B[M-1][0]=bt/(hx**2)
        

BB[0:M,0:M],BB[M:2*M,M:2*M]=B,B
BB[2*M:3*M,2*M:3*M]=B



def fun1(x,z):
    b=np.zeros((3*M,1))
    u=z[0:M]
    v=z[M:2*M]
    w=z[2*M:3*M]
    
    for j in range(0,M):
        b[j]=(-1/bf)*(u[j]**3+u[j]*v[j]+w[j])
        b[M+j]=0.07*(u[j]-0.7)*(u[j]-1.3)/((u[j]-0.7)*(u[j]-1.3)+0.1)+w[j]
        b[2*M+j]=-((v[j])**2)*w[j]+0.035*(u[j]-0.7)*(u[j]-1.3)/((u[j]-0.7)*(u[j]-1.3)+0.1)+w[j]-v[j]-0.4*u[j]
    b=b.reshape((3*M,1))
    U=np.dot(BB,z).reshape((3*M,1))
    return U+b
def err(x,y,tc,h):
    x1=x.reshape((3*M,1))
    y1=y.reshape((3*M,1))
    z1=12*(x1-y1)
    return (0.1) *(z1+6*h*(fun1(tc+h,x1)+fun1(tc+h,y1)))

eig1,abcd=np.linalg.eig(BB)
eig2=np.max(np.abs(eig1)) 

def ro(x,y):
    e=1e-8;ln=len(y)
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
    while fg > 1e-3*R and fg1<20:
        Rv1=Rv+e*(f1-f2)/np.linalg.norm(f1-f2)
        f1=fun1(x,Rv1)
        R=np.linalg.norm(f1-f2)/e
        fg=np.abs(R-Rr)
        fg1+=1
        Rr=R 
    if fg1==20:
        R=1.1*R
    return R,fg1

RKCv2= np.load(r'C:\Users\A204-7\Desktop\RKC\RKC\RKC2.npz', allow_pickle=True)
cs = RKCv2['cs']
us1=RKCv2['us1']
vs1=RKCv2['vs1']
vs=RKCv2['vs']
us=RKCv2['us']



def RKC(fun1,t0,t_end,h,u0,s): 
    h1=h
    tc=[t0] #t的初始
    y=u0
    counter=0
    fg1=0
    nfe=0
    s_max=0
   
    while tc[-1]<t_end:
        c=cs[s,0]
        u1=us1[s,0]
        u=us[s,0]
        v1=vs1[s,0]
        v=vs[s,0]
       
        nfe=s+nfe+fg1+3
        k0=np.zeros((3*M,1))
        k1=np.zeros((3*M,1))
        k2=np.zeros((3*M,1))
        k3=np.zeros((3*M,1))
        ky0=np.zeros((3*M,1))
        ky1=np.zeros((3*M,1))
        k0=y[:,-1].copy()
        k0=k0.reshape((3*M,1))
        ky0=fun1(tc[-1],k0)
        k1=k0+u1[1] *h *ky0
        
        ky1=fun1(tc[-1]+u1[1]*h,k1) 
        k2=k1.copy()
        k1=k0.copy()
        for j in range(2,s+1):

            k3=u[j]*k2+v[j]*k1+(1-u[j]-v[j])*k0+u1[j]*h*ky1+v1[j]*h*ky0
            #if j==4:
                #print(k[4])
            ky1=fun1(tc[-1]+c[j]*h,k3)
            k1=k2.copy()
            k2=k3.copy()
      
        
        yc=k3.copy()
            #err2=err(y[:,-1],yc,tc[-1],h1)
            #err1=np.linalg.norm(err2)/math.sqrt(2*M+2)
            #print(err1)
            # fac=0.8*((1/err1)**(1/3))
      
        y =yc.copy()
        
        counter+=1
        pu,fg1=ro(tc[-1]+h1,yc)
        tc.append(tc[-1]+h1)
        if    h1 > t_end-tc[-1]:
                 h1 = t_end -tc[-1]
                 h=h1
        s2=math.sqrt(h1*pu/0.55)
        s=math.ceil(s2)
        if s_max<s:
                   s_max=s
        if s<2:
                    s=2
        if s>250:
                s=250  
    
         
            
    return np.array(tc),np.array(y),nfe,s_max
t0=0
t_end=1.1
h=0.005
eig3,fg1=ro(0,y)

s2=np.sqrt(h*eig3/0.55)                                           
s=math.ceil(s2)

print('eig:',eig3)
eig1,abcd=np.linalg.eig(BB)
eig2=np.max(np.abs(eig1))
print('eig2:',eig2)
print(ro(0,y))
#print(fun1(0,y))
#print(y)
if s<=1:
    s=2
#tc1,y1,nfe1,s_max1=RKC2(fun1,t0,t_end,0.0001,y,s)
tc,y,nfe,s_max=RKC(fun1,t0,t_end,h,y,s)
 
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



#solu1=np.load('Robertsoneqsolu.npy')
#err=sum([(x - y) ** 2 for x, y in zip(y[1:3*M-3,-1], solu[1:3*M-3])] )/ len(solu[1:3*M+2])
#print("err:",np.sqrt(err))
Robsolu=y.ravel()
#print(Robsolu)

# 创建第二组数据

#df = pd.DataFrame({'cuspsolu_0.000001': Robsolu})

# 保存到新的 Excel 文件

#df.to_excel("SERKv2ROBsolu.xlsx", index=False)
#np.save('SERKV2ROBsolu0.000001.npy',Robsolu)
#solu1=np.load('SERKV2ROBsolu0.000001.npy')
#solu1=np.load(r'C:\Users\A204-7\Desktop\RKC\RKC\Robertsoneqsolu.npy')

#err=sum([(x - y) ** 2 for x, y in zip(y[0:3*M-3,-1], solu[0:3*M-3])] )/ len(solu[0:3*M-3])
#print("err:",np.sqrt(err))