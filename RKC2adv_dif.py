import numpy as np   #rkc2定步长 Robertson equation
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import copy
import math
import pandas as pd



np.seterr(divide='ignore', invalid='ignore')
pi=math.pi
M=600
time_st=time.time()
x0=-pi
x_end=pi
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
bt=0.05
af=0.1
A=np.zeros((M-1,M-1)) 
A1=np.zeros((M-1,M-1))
A2=np.zeros((M-1,M-1))
solu=np.zeros((M+1,1))
y=np.zeros((M-1,1))
a1=bt/(hx**2)
b1=-2*bt/(hx**2)
c1=bt/(hx**2)
a2=(-3*af)/(2*hx)
b2=(2*af)/hx
c2=-af/(2*hx)





'''
A[0][0],A[0][1],A[0][2],A[0][3]=2*bt/(hx**2)+af/hx,-5*bt/(hx**2)-af/hx,4*bt/(hx**2),-bt/(hx**2)
A[1][0],A[1][1],A[1][2]=bt/(hx**2)+af/hx,-2*bt/(hx**2)-af/hx,bt/(hx**2)
for i in range(M+1):
    y[i]=math.sin(x[i])
    solu[i]=np.e**(-bt*2)*np.sin(x[i]-af*2)
    if i>=2 and i<=M-1:
        A[i][i-2],A[i][i-1]=-af/(2*hx),bt/(hx**2)+4*af/(2*hx)
        A[i][i],A[i][i+1]=-2*bt/(hx**2)-3*af/(2*hx),bt/(hx**2)
    if i==M:
        A[M][1],A[M][M-2]=bt/(hx**2),-af/(2*hx)
        A[M][M-1],A[M][M]=bt/(hx**2)+4*af/(2*hx),-2*bt/(hx**2)-3*af/hx
      
          
'''
for i in range(M+1):
     if i==0:
          A1[0][0]=b1
          A1[0][1]=c1 
          A2[0][0]=a2*2/3
          solu[i]=np.e**(-bt*2)*np.sin(x[i]-af*2)
          y[i]=math.sin(x[i+1])
     elif i==1:
          A1[i][i-1]=a1
          A1[i][i]=b1
          A1[i][i+1]=c1
          A2[i][i-1]=b2
          A2[i][i]=a2
          solu[i]=np.e**(-bt*2)*np.sin(x[i]-af*2)
          y[i]=math.sin(x[i+1])
     elif i>1 and i<=M-3 :
          A1[i][i-1]=a1
          A1[i][i]=b1
          A1[i][i+1]=c1
          A2[i][i-1]=b2
          A2[i][i]=a2
          A2[i][i-2]=c2
          solu[i]=np.e**(-bt*2)*np.sin(x[i]-af*2)
          y[i]=math.sin(x[i+1])
     elif i==M-2:
        A1[i][i-1]=a1
        A1[i][i]=b1
        A2[i][i-1]=b2
        A2[i][i]=a2
        A2[i][i-2]=c2
        solu[i]=np.e**(-bt*2)*np.sin(x[i]-af*2)
        y[i]=math.sin(x[i+1])
     else:
        solu[i]=np.e**(-bt*2)*np.sin(x[i]-af*2)
A=A1+A2
print(A)

def fun1(x,z):
    U=np.dot(A,z).reshape((M-1,1))
    return U



def err(x,y,tc,h):
    x1=x.reshape((M+1,1))
    y1=y.reshape((M+1,1))
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
       
        nfe=s+nfe
        k0=np.zeros((M-1,1))
        k1=np.zeros((M-1,1))
        k2=np.zeros((M-1,1))
        k3=np.zeros((M-1,1))
        ky0=np.zeros((M-1,1))
        ky1=np.zeros((M-1,1))
        k0=y[:,-1].copy()
        k0=k0.reshape((M-1,1))
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
        y = np.column_stack((y, yc))
        counter+=1
        tc.append(tc[-1]+h1)
        #pu,fg1=ro(tc[-1]+h1,yc)
        
        
         
            
    return np.array(tc),np.array(y),nfe,s_max
t0=0
t_end=2
h=0.001
eig1,abcd=np.linalg.eig(A)
eig2=np.max(np.abs(eig1)) 
print("eig2:",eig2)                                       
s2=math.sqrt(h*eig2/0.55)
s=math.ceil(s2)

#print(fun1(x,y))
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
err=sum([(x - y) ** 2 for x, y in zip(y[:,-1], solu[1:M])] )/ len(solu[1:M])
print("err:",np.sqrt(err))
plt.plot(x[1:M], y[:,-1],'red')
plt.plot(x[1:M], solu[1:M],'blue')
plt.title(' t=2 af=0.1 beta=0.05  numberical solutions of RKC')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
'''
df = pd.read_excel("solu.xlsx")

# 创建第二组数据

# 将第二组数据放在第二列
df["RObsolual3rd_0.0001  "] = Robsolu

# 保存到Excel文件
df.to_excel("solu.xlsx", index=False)
np.save('Robertsoneqsolu.npy',Robsolu)'''