import numpy as np   #改造
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import math
np.seterr(divide='ignore', invalid='ignore')
M=200
time_st=time.time()
x0=0
x_end=1
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
bt=0.04
bf=0.001
e=np.zeros((M+1,1))
BB=np.zeros((3*M+3,3*M+3))
B=np.zeros((M+1,M+1)) 
y=np.zeros((3*M+3,1))
tol=1e-3    
for i in range(0,M+1):
    if i==0:
        y[i]=0
        y[M+1+i]=-2*np.cos(2*np.pi*x[i])
        y[2*M+2+i]=2*np.sin(2*np.pi*x[i])
        e[0]=0
        B[0][0],B[0][1]=-2*bt/(hx**2),bt/(hx**2)
        B[0][M-1]=bt/(hx**2)
    elif 0<i<M:
        y[i]=0
        y[M+1+i]=-2*np.cos(2*np.pi*x[i])
        y[2*M+2+i]=2*np.sin(2*np.pi*x[i])
        e[i]=1
        B[i][i-1],B[i][i],B[i][i+1]=bt/(hx**2),-2*bt/(hx**2),bt/(hx**2)
    else:
        y[i]=0
        y[M+1+i]=-2*np.cos(2*np.pi*x[i])
        y[2*M+2+i]=2*np.sin(2*np.pi*x[i])
        e[i]=0
        B[M][M],B[M][M-1]=-2*bt/(hx**2),bt/(hx**2)
        B[M][1]=bt/(hx**2)

BB[0:M+1,0:M+1],BB[M+1:2*M+2,M+1:2*M+2]=B,B
BB[2*M+2:3*M+3,2*M+2:3*M+3]=B+np.eye(M+1)
BB[0:M+1,2*M+2:3*M+3]=(-1/bf)*np.eye(M+1)
BB[M+1:2*M+2,2*M+2:3*M+3]=np.eye(M+1)
BB[2*M+2:3*M+3,0:M+1],BB[2*M+2:3*M+3,M+1:2*M+2]=-0.4*np.eye(M+1),-1*np.eye(M+1)


def err(x,y,h):
    return (1/15)*(12*(x-y)+6*h*(np.dot(BB,x)+np.dot(BB,y)))
def fun1(x,y):
    b=np.zeros((3*M+3,1))
    u=y[0:M+1]
    v=y[M+1:2*M+2]
    w=y[2*M+2:3*M+3]
    
    for j in range(0,M+1):
        b[j]=-(u[j]**3+u[j]*v[j])/bf
        b[M+1+j]=0.07*(u[j]-0.7)*(u[j]-1.3)/((u[j]-0.7)*(u[j]-1.3)+0.1)
        b[2*M+2+j]=-(v[j]**2)*w[j]+0.035*(u[j]-0.7)*(u[j]-1.3)/((u[j]-0.7)*(u[j]-1.3)+0.1)
  
    '''
    u=np.array(y[0:M+1])
    w=np.array(y[M+1:2*M+2])
    v=np.array(y[2*M+2:3*M+3])
    u=u.reshape((201,1))
    w=w.reshape((201,1))
    v=v.reshape((201,1))
    #b[0:M+1]=-(u**3+u*v)/bf
    #b[M+1:2*M+2]=0.07*(u-0.07*e)*(u-0.13*e)/((u-0.7*e)*(u-1.3*e)+0.1*e)
    #b[2*M+2:3*M+3]=-(v**2)*w-v-0.4*u+0.035*(u-0.07*e)*(u-0.13*e)/((u-0.7*e)*(u-1.3*e)+0.1*e)
    u1=-(u**3+u*v)/bf
    v1=0.07*(u-0.07*e)*(u-0.13*e)/((u-0.7*e)*(u-1.3*e)+0.1*e)
    w1=-(v**2)*w-v-0.4*u+0.035*(u-0.07*e)*(u-0.13*e)/((u-0.7*e)*(u-1.3*e)+0.1*e)
    b=np.vstack((u1, v1, w1))'''
    b=b.reshape((603,1))
    U=np.dot(BB,y).reshape((3*M+3,1))
    uuu=U+b
    uuu=uuu.reshape((603,))
    return uuu
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
        k=np.zeros((3*M+3,4))
        ky=np.zeros((3*M+3,2))
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
        tc.append(tc[-1]+h)
        if counter==0:
            yc=k[:,3]
            counter+=1
        else :
            yc=k[:,3]
        y = np.column_stack((y, yc))
    return np.array(tc),np.array(y)
t0=0
t_end=2
h=0.01
eig1,abcd=np.linalg.eig(BB)
eig2=np.max(np.abs(eig1))
print(eig2)
s2=math.sqrt(h*eig2/0.45)
s=math.ceil(s2)

print(fun1(x,y))
if s<3:
    s=2
tc,y=RKC(fun1,t0,t_end,h,y,s)
#mse = np.mean((np.array(y[1:M,-1]) - np.array(solu[1:M]))**2)
#mae = np.mean(np.abs(np.array(y[1:M,-1]) - np.array(solu[1:M])
# ))
time_end=time.time()
err2=err(y[:,-2],y[:,-1],h)
err1=np.linalg.norm(err2)
print(err1)
'''
print(time_end-time_st)
plt.plot(x, y[:,-1],'red')
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
#plt.show()        
'''