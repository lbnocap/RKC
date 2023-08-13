import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
from numpy.polynomial import chebyshev
np.seterr(divide='ignore', invalid='ignore')

M=600
x0=-math.pi
x_end=math.pi
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
A=np.zeros((M+1,M+1))
y=np.zeros((M+1,1))
bt=0.1
af=0
tol=1e-3
A=np.zeros((M+1,M+1))
'''
A[0][0],A[0][1]=-2*bt/(hx**2)-3*af/(2*hx),bt/(hx**2)
A[0][M-1],A[0][M-2]=bt/(hx**2)+4*af/2*hx,-af/(2*hx)
A[1][0],A[1][1],A[1][2]=bt/(hx**2)+4*af/(2*hx),-2*bt/(hx**2)-3*af/(2*hx),bt/(hx**2)
A[1][M-1]=-af/(2*hx)
'''
A[0][0],A[0][1],A[0][2],A[0][3]=2*bt/(hx**2)+af/hx,-5*bt/(hx**2)-af/hx,4*bt/(hx**2),-bt/(hx**2)
A[1][0],A[1][1],A[1][2]=bt/(hx**2)+af/hx,-2*bt/(hx**2)-af/hx,bt/(hx**2)

for i in range(M+1):
    y[i]=math.sin(x[i])
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
eig1,abcd =np.linalg.eig(A)
eig2=np.max(np.abs(eig1)) 
B=A[1:M,1:M]   
def fun1(x,y):
     return np.dot(A,y)

def err(x,y,h):
    return (1/15)*(12*(x-y)-6*h*(np.dot(A,x)+np.dot(A,y)))
def RKC(f,t0,t_end,h,u0,s):
    h_v=[h]
    h_v2=[h]
    h1=h
    tc=[t0] #t的初始
    y=u0
    s_max=1
    counter=0
    cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
    cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
    t3=cheb_poly.deriv(1)
    t4=cheb_poly.deriv(2)
    t5=cheb_poly.deriv(3)
    r=1
    while tc[-1]<t_end:
        yt=0.6
        w0=1+0.05/((s)**2)
        c=np.zeros(s+1)
        c1=np.zeros(s+1)
        b=np.zeros(s+1)
        t=np.zeros(s+1)
        t1=np.zeros(s+1)
        k=np.zeros((M+1,s+1))
        k1=np.zeros((M+1,s+1))
        ky=np.zeros((M+1,s+1))
        x=np.zeros(s+1)
        x[0],x[1]=0,0 
        ky[:,0]=y[:,-1]
   
        c1[0]=0
        c[0]=0
        t1[0]=0
        t1[1]=1
        b[0]=1
        t[0]=1
        t[1]=w0
        b[1]=1/t[1]
        u=np.zeros(s+1)
        u[0],u[1]=0,0
        v=np.zeros(s+1)
        v[0],v[1]=0,0  
        u1=np.zeros(s+1)
    
        for j in range(2,s+1):
         t[j]=2*w0*t[j-1]-t[j-2]
         t1[j]=2*t[j-1]+2*w0*t1[j-1]-t1[j-2]
         b[j]=1/t[j]  
    #print(b)  
        w1=1/(0.341*(s**2))
        u1[0],u1[1]=0,w1/w0
        k[:,0],k1[:,0]=y[:,-1],y[:,-1]
        ky[:,1]=fun1(t[-1],y[:,-1])
        k[:,1]=y[:,-1]+(w1/w0) *h *ky[:,1]
        c[1]=w1/w0
        for j in range(2,s+1):
            u[j]=2*w0*b[j]/b[j-1]
            #print(u[j])
            v[j]=-b[j]/b[j-2]
            #print(v[j])
            u1[j]=2*w1*b[j]/b[j-1]
            c[j]=u[j]*c[j-1]+v[j]*c[j-2]+u1[j]
            x[j]=u[j]*x[j-1]+v[j]*x[j-2]+u1[j]*c[j-1]
            k[:,j]=u[j]*k[:,j-1]+v[j]*k[:,j-2]+(1-u[j]-v[j])*k[:,0]+u1[j]*h*ky[:,j-1]
            #if j==4:
                #print(k[4])
            ky[:,j]=fun1(tc[-1]+c[j]*h,k[:,j])
        bb=cheb_poly(w0)
        bs=bb/(t3(w0)*w1)
        bn=(1+r)/(bs*(yt*c[s]+2*x[s]*r*(yt**2)))
        bf1=(c[s]*r*(1+r))/(c[s]+2*x[s]*r*yt)-r
        b0=1-bf1-bn
        yb=k[:,s]
        err11=0
        err1=0
        if counter>=1:
         yc=bf1*y[:,-2]+b0*y[:,-1]+bn*k[:,s]
         err22=err(y[:,-1],yc,h1)/tol
         err11=np.linalg.norm(err22)/math.sqrt(M+1)
         fac2=0.8*((1/err11)**(1/2))
        if counter==0:
         err2=err(y[:,-1],yb,h1)/tol
         err1=np.linalg.norm(err2)/math.sqrt(M+1)
         fac=0.8*((1/err1)**(1/2))
        if counter==0 and err1<1:
            y=y = np.column_stack((y, yb))
            h_v.append(h1)
            h_v2.append(h1)
            tc.append(tc[-1]+h1)
            h1=min(h1,fac*h1)
            r=h1/h
            s=math.sqrt(h*eig2/0.65)
            s=math.ceil(s)
            counter+=1
            h=yt*h1
            if s<=2:
                s=3
            if s>200:
                s=200
            if s>s_max:
                s_max=s
        if counter==0 and err1>1:
            h=fac*h
            s=math.sqrt(h*eig2/1.138)
            s=math.ceil(s)
            if s<=2:
                s==3
            if s>200:
                s=200
        if err11<1 and counter>=1:
            y = np.column_stack((y, yb))
            h_v.append(h1)
            h_v2.append(h1)
            h1=min(10,max(0.1,fac))*h1
            if tc[-1] + h1 > t_end:
                 h1 = t_end -tc[-1]
            s=math.sqrt(h1*eig2/1.137)
            s=math.ceil(s)
            tc.append(tc[-1]+h1)
            if s==1:
                s==2
            if s>200:
                s=200
            if s>s_max:
                s_max=s
            h=yt*h1
        if err11>1 and counter >=1:
            h1=min(10,max(0.1,fac))*h1
            h_v.append(h1)
            r=h_v[-1]/h_v[-1]
            h_v2.append(h1)
            s=math.sqrt(h*eig2/1.137)
            s=math.ceil(s)
            if s==1:
                s==2
            if s>200:
                s=200
            h=yt*h1
            
            

    err3=err(y[:,-2],y[:,-1],h_v2[-1])
    err31=np.linalg.norm(err3)/math.sqrt(M+1)
    #print(err31)
    print(s_max)
    #print(h_v2)
    return np.array(tc),np.array(y)

t0=0
t_end=2
h=0.001
eig11,abcd=np.linalg.eig(B)
eig21=np.max(np.abs(eig1))
print(eig21)
s=math.sqrt(h*eig2/1.0)
s=math.ceil(s)
tc,y=RKC(fun1,t0,t_end,h,y,s)
plt.plot(x[1:M], y[1:M,-1],)
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