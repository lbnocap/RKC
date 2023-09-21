import numpy as np   #原二步定步长非线性例三
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import copy
import math
import h5py

pi=math.pi
M=1000
time_st=time.time()
x0=0
x_end=1
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
bt=1/(25* pi**2)
af=26/(25* (pi**2))
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





def RKC(f,t0,t_end,h,u0,s): 
    h1=h
    tc=[t0] #t的初始
    y=u0
    counter=0
    fg1=0
    nfe=0
    s_max=0
    lop=0
    while tc[-1]<t_end:
        nfe=s+nfe+fg1+3
        cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
        cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
        t3=cheb_poly.deriv(1)
        t4=cheb_poly.deriv(2)
        t5=cheb_poly.deriv(3)
        t42=chebyshev.Chebyshev([0] * (2 + 1))
        t42.coef[-1]=1
        t22=t42.deriv(2) 
        w0=1+(0.9)/((s)**2)
        c=np.zeros(s+1)
        b=np.zeros(s+1)
        t=np.zeros(s+1)
        t1=np.zeros(s+1)
        k0=np.zeros((3*M+3,1))
        k1=np.zeros((3*M+3,1))
        k2=np.zeros((3*M+3,1))
        k3=np.zeros((3*M+3,1))
        ky0=np.zeros((3*M+3,1))
        ky1=np.zeros((3*M+3,1))
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
        k0=y[:,-1].copy()
        k0=k0.reshape((3*M+3,1))
        ky0=fun1(t[-1],k0)
        k1=k0+u1[1] *h *ky0
        ky1=fun1(t[-1]+u1[1]*h,k1)
        c[1]=u1[1]
        k2=k1.copy()
        k1=k0.copy()
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
            k3=u[j]*k2+v[j]*k1+(1-u[j]-v[j])*k0+u1[j]*h*ky1+v1[j]*h*ky0
            ky1=fun1(tc[-1]+c[j]*h,k3)
            if j==10:
                print(u[j],v1[j])
            k1=k2.copy()
            k2=k3.copy()
       
     
        r=1
      
        
        cc=t3(w0)*t5(w0)/(t4(w0)**2)
        yt=1/np.sqrt(cc)
        #yt=0.8
        bn=(1+r)/(yt*(1+r*yt))
        bf1=(r**2)*(1-yt)/(1+yt*r)
        b0=1-bn-bf1
        C=1/6+bf1/6-bn*(yt**3)*cc/6
        
        if counter==0:
            yc=k3.copy()
            #print(yc)
            #err2=err(y[:,-1],yc,tc[-1],h1)
            #err1=np.linalg.norm(err2)/math.sqrt(3*M+3)
            #print(yc)
            #fac=0.8*((1/err1)**(1/3))
            y = np.column_stack((y, yc))
            counter=1
            tc.append(tc[-1]+h1) 
           
            pu,fg1=ro(tc[-1]+h1,yc)
            if pu>lop:
                lop=pu
            s2=math.sqrt(h1*pu/0.4)
            s=math.ceil(s2)
            if s_max<s:
                   s_max=s
            if s<3:
                    s=3
           # h=yt*h1
            


           
        else :
            k02=y[:,-2].copy()
            k02=k02.reshape((3*M+3,1))
            yb=k3.copy()
            yc=bf1*k02+b0*k0+bn*yb
            pu,fg1=ro(tc[-1]+h1,yc)
            tc.append(tc[-1]+h1)
            if tc[-1] + h1 > t_end:
                 h1 = t_end -tc[-1]
    
            s2=np.sqrt(h1*pu/0.4)                                           
            s=math.ceil(s2)
            if pu>lop:
                lop=pu
            if s<3:
                s=3
            if s>s_max:
                s_max=s
            h=yt*h1
            #h=h1
            y = np.column_stack((y, yc))

    return np.array(tc),np.array(y),nfe,s_max,lop
t0=0
t_end=1
h=0.01
eig3,fg1=ro(0,y)
s2=np.sqrt(h*eig3/0.4)                                           
s=int(s2)
print('eig:',eig3)
#print(fun1(x,y))
#print(y)
if s<=3:
    s=3
tc,y,nfe,s_max,lop=RKC(fun1,t0,t_end,h,y,s)

#mse = np.mean((np.array(y[1:M,-1]) - np.array(solu[1:M]))**2)
#mae = np.mean(np.abs(np.array(y[1:M,-1]) - np.array(solu[1:M])
# ))
#err=sum([(x - y) ** 2 for x, y in zip(y[1:M,-1], solu[1:M])] )/ len(solu[1:M])
#rint(np.sqrt(err))
time_end=time.time()
print("lop:",lop)
print(time_end-time_st)
print(tc)
print("步数：",len(tc))
print("评估次数：",nfe)
print("s_max:",s_max)
err2=err(y[:,-3],y[:,-2],tc[-3],h)
print(y[:,-1])
print(solu)
err1=np.linalg.norm(err2)/math.sqrt(3*M+3)
print("err:",err1)
with h5py.File('eig3solution.h5', 'r') as hf:
    solu = hf['solu1'][:]
    
err=sum([(x - y) ** 2 for x, y in zip(y[1:3*M+3,-1], solu[1:3*M+3])] )/ len(solu[1:3*M+3])
print("err:",np.sqrt(err))

#plt.plot(x, y[:,-1],'red')
#plt.plot(x, solu,'blue')
#plt.title(' t=2 af=0.1 beta=0.05  numberical solutions of RKC')
#plt.xlabel('x')
#plt.ylabel('y') 
#plt.legend()
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#X, Y = np.meshgrid(x, tc)
#ax.plot_surface(X,Y,y.T, rstride=1, cstride=1, cmap='hot')
#plt.title('3D numberical solutions of RKC')

