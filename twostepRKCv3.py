import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
from numpy.polynomial import chebyshev
import time
np.seterr(divide='ignore', invalid='ignore')
M=200
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
tol=1e-3
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
def err(x,y,h):
    return (1/15)*(12*(x-y)-6*h*(np.dot(A,x)+np.dot(A,y)))
def fun1(x,y):
     return np.dot(A,y)
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
        w0=1+(0.05)/((s)**2)
        c=np.zeros(s+1)
        b=np.zeros(s+1)
        t=np.zeros(s+1)
        t1=np.zeros(s+1)
        k=np.zeros((M+1,4))
        ky=np.zeros((M+1,2))
        ky[:,0]=y[:,-1]
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
        if tc[-1] + h > t_end:
            h = t_end -tc[-1]
        k[:,0]=y[:,-1]
        ky[:,0]=fun1(t[-1],y[:,-1])
        k[:,1]=y[:,-1]+u1[1] *h *ky[:,0]
        ky[:,1]=fun1(t[-1]+u1[1]*h,k[:,1])
        c[1]=u1[1]
        k[:,2]=k[:,1]
        k[:,1]=k[:,0]
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
            c[j]=u[j]*c[j-1]+v[j]*c[j-2]+u1[j]
            v1[j]=-(1-b[j-1]*t[j-1])*u1[j]
            k[:,3]=u[j]*k[:,2]+v[j]*k[:,1]+(1-u[j]-v[j])*k[:,0]+u1[j]*h*ky[:,1]+v1[j]*h*ky[:,0]
            #if j==4:
                #print(k[4])
            ky[:,1]=fun1(tc[-1]+c[j]*h,k[:,3])
            k[:,1]=k[:,2]
            k[:,2]=k[:,3]
        r=1
        cc=t3(w0)*t5(w0)/(t4(w0)**2)
        #yt=1/np.sqrt(cc)
        yt=0.6
        bn=(1+r)/(yt*(1+r*yt))
        bf1=(r**2)*(1-yt)/(1+yt*r)
        b0=1-bn-bf1
        h_v.append(h1)
        tc.append(tc[-1]+h1)
        if counter==0:
            yc=k[:,3]
            counter+=1
            h=yt*h
        else :
            yc=bf1*y[:,-2]+b0*y[:,-1]+bn*k[:,3]
        y = np.column_stack((y, yc))
    return np.array(tc),np.array(y)
t0=0
t_end=2
h=0.1
eig1,abcd=np.linalg.eig(A)
eig2=np.max(np.abs(eig1))
print(eig2)
s2=math.sqrt(h*eig2/1.0)
s=math.ceil(s2)
print(s)
tc,y=RKC(fun1,t0,t_end,h,y,s)
#mse = np.mean((np.array(y[1:M,-1]) - np.array(solu[1:M]))**2)
#mae = np.mean(np.abs(np.array(y[1:M,-1]) - np.array(solu[1:M])
# ))
err=sum([(x - y) ** 2 for x, y in zip(y[1:M,-1], solu[1:M])] )/ len(solu[1:M])
print(np.sqrt(err))
yyy=y[1:M,-1].reshape(M-1,1)
err2=np.linalg.norm(yyy-solu[1:M])/np.sqrt(M-1)
print(err2)
time_end=time.time()
print(time_end-time_st)
#plt.plot(x, y[:,-1],'red')
plt.plot(x, solu,'blue')
plt.title(' t=2 af=0.1 beta=0.05  numberical solutions of RKC')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(x, tc)
ax.plot_surface(X,Y,y.T, rstride=1, cstride=1, cmap='hot')
plt.title('3D numberical solutions of RKC')
plt.show()        