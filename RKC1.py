import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
np.seterr(divide='ignore', invalid='ignore')

M=600
x0=-math.pi
x_end=math.pi
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
A=np.zeros((M+1,M+1))
y=np.zeros((M+1,1))
bt=0.05
af=0.1 
A=np.zeros((M+1,M+1))
tol=1e-5
err1=[0]
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
B=A[1:M,1:M]
def err(x,y,h):
    return (1/15)*(12*(x-y)-6*h*(np.dot(A,x)+np.dot(A,y)))
def fun1(x,y):
     return np.dot(A,y)
def RKC(f,t0,t_end,h,u0,s,s1):
    h_v=[h]
    tc=[t0] #t的初始
    y=u0
    '''
    w0=1+0.05/((s+s1)**2)
    c=np.zeros(s+s1+1)
    c1=np.zeros(s+s1+1)
    b=np.zeros(s+s1+1)
    t=np.zeros(s+1+s1)
    t1=np.zeros(s+1+s1)
    k=np.zeros((M+1,s+s1+1))
    k1=np.zeros((M+1,s+1+s1))
    ky=np.zeros((M+1,s+s1+1))
    ky[:,0]=y[:,-1]
   
    c1[0]=0
    c[0]=0
    t1[0]=0
    t1[1]=1
    b[0]=1
    t[0]=1
    t[1]=w0
    b[1]=1/t[1]
    u=np.zeros(s+1+s1)
    u[0],u[1]=0,0
    v=np.zeros(s+1+s1)
    v[0],v[1]=0,0  
    u1=np.zeros(s+1+s1)
    
    for j in range(2,s+s1+1):
        t[j]=2*w0*t[j-1]-t[j-2]
        t1[j]=2*t[j-1]+2*w0*t1[j-1]-t1[j-2]
        b[j]=1/t[j]  
    #print(b)  
    w1=t[s+s1]/t1[s+s1]
    u1[0],u1[1]=0,w1/w0 
    '''
    while tc[-1]<t_end:
        w0=1+0.05/((s+s1)**2)
        c=np.zeros(s+s1+1)
        c1=np.zeros(s+s1+1)
        b=np.zeros(s+s1+1)
        t=np.zeros(s+1+s1)
        t1=np.zeros(s+1+s1)
        k=np.zeros((M+1,s+s1+1))
        k1=np.zeros((M+1,s+1+s1))
        ky=np.zeros((M+1,s+s1+1))
        ky[:,0]=y[:,-1]

        c1[0]=0
        c[0]=0
        t1[0]=0
        t1[1]=1
        b[0]=1
        t[0]=1
        t[1]=w0
        b[1]=1/t[1]
        u=np.zeros(s+1+s1)
        u[0],u[1]=0,0
        v=np.zeros(s+1+s1)
        v[0],v[1]=0,0  
        u1=np.zeros(s+1+s1)
    
        for j in range(2,s+s1+1):
         t[j]=2*w0*t[j-1]-t[j-2]
         t1[j]=2*t[j-1]+2*w0*t1[j-1]-t1[j-2]
         b[j]=1/t[j]  
        
    #print(b)  
        w1=t[s+s1]/t1[s+s1]
        u1[0],u1[1]=0,w1/w0 
        h=h_v[-1]
        if tc[-1] + h > t_end:
            h = t_end -tc[-1]
        k[:,0],k1[:,0]=y[:,-1],y[:,-1]
        ky[:,1]=fun1(t[-1],y[:,-1])
        k[:,1]=y[:,-1]+(w1/w0) *h *ky[:,1]
        c[1]=w1/w0
        c1[1]=1/(s1+1)*c[1]
        k1[:,1]=1/(s1+1)*k[:,1]+(1-1/(s1+1))*k1[:,0]
        for j in range(2,s+s1+1):
            u[j]=2*w0*b[j]/b[j-1]
            #print(u[j])
            v[j]=-b[j]/b[j-2]
            #print(v[j])
            u1[j]=2*w1*b[j]/b[j-1]
            c[j]=u[j]*c[j-1]+v[j]*c[j-2]+u1[j]
            c1[j]=1/(s1+1)*c[j]+(1-1/(s1+1))*c1[j-1]
            k[:,j]=u[j]*k[:,j-1]+v[j]*k[:,j-2]+(1-u[j]-v[j])*k[:,0]+u1[j]*h*ky[:,j-1]
            #if j==4:
                #print(k[4])
            k1[:,j]=1/(s1+1)*k[:,j]+(1-1/(s1+1))*k1[:,j-1]
            ky[:,j]=fun1(tc[-1]+c[j]*h,k[:,j])
        x1=(1-c1[s+s1])/(c1[s+s1-1]-c1[s+s1])
        x2=1-x1
        h_v.append(h)
        tc.append(tc[-1]+h)
        yb=x1*k1[:,s+s1-1]+x2*k1[:,s+s1]
        yc=k[:,s+s1]
        err2=err(y[:,-1],yc,h)/tol
        err1.append(np.linalg.norm(err2)/math.sqrt(M+1))
        fac=0.8*((1/err1[-1])**(1/2))
        y = np.column_stack((y, yb))
        h3=(min(10,max(0.1,fac)))*h
        eig1,abcd=np.linalg.eig(A)
        eig2=np.max(np.abs(eig1))
        s2=math.sqrt(h3*eig2/0.65)
        s2=math.ceil(s2)
    return np.array(tc),np.array(y)

t0=0
t_end=2
h=0.01
tc,y=RKC(fun1,t0,t_end,h,y,50,2)
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
eig1b,abcd=np.linalg.eig(B)
eig2b=np.max(np.abs(eig1b))
print(eig2b)
plt.show()






