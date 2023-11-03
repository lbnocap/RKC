import numpy as np   #原二阶三步非线性例5
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
af=2
gm=-2000
e=np.zeros((M-1,1))
A=np.zeros((2*M-2,M*2-2))
B1=np.zeros((M-1,M-1)) 
B2=np.zeros((M-1,M-1))
y=np.zeros((2*M-2,1))
solu=np.zeros((2*M-2,1))
tol=1e-3 
for i in range(0,M-1):
    if i==0:
        solu[i]=np.exp(-((pi)**2) *bt*1)*np.sin(pi*(x[i+1]-1))
        solu[M-1+i]=np.exp(-((pi)**2) *bt*1)*np.cos(pi*(x[i+1]-1))
        y[i]=np.sin(pi*x[i+1])
        y[M-1+i]=np.cos(pi*x[i+1])
        B1[0][0],B1[0][1]=-2*bt/(hx**2)+gm,bt/(hx**2)
        B1[M-2][M-2],B1[M-2][M-3]=-2*bt/(hx**2)+gm,bt/(hx**2),
        B2[0][0]=-af/hx
        
    elif 0<i<M-2:
        solu[i]=np.exp(-((pi)**2) *bt*1)*np.sin(pi*(x[i+1]-1))
        solu[M-1+i]=np.exp(-((pi)**2) *bt*1)*np.cos(pi*(x[i+1]-1))
        y[i]=np.sin(pi*x[i+1])
        y[M-1+i]=np.cos(pi*x[i+1])
        if i==1:
            B1[i][i-1],B1[i][i],B1[i][i+1]=bt/(hx**2),-2*bt/(hx**2)+gm,bt/(hx**2)
            B2[i][i-1],B2[i][i]=2*af/hx,-(3*af)/(2*hx)
        else:
           B1[i][i-1],B1[i][i],B1[i][i+1]=bt/(hx**2),-2*bt/(hx**2)+gm,bt/(hx**2)
           B2[i][i-1],B2[i][i]=2*af/hx,-(3*af)/(2*hx)
           B2[i][i-2]=-af/(2*hx)
       
    elif i==M-2:
        solu[i]=np.exp(-((pi)**2) *bt*1)*np.sin(pi*(x[i+1]-1))
        solu[M-1+i]=np.exp(-((pi)**2) *bt*1)*np.cos(pi*(x[i+1]-1))
        y[i]=np.sin(pi*x[i+1])
        y[M-1+i]=np.cos(pi*x[i+1])
        B2[i][i-1],B2[i][i]=2*af/hx,-(3*af)/(2*hx)
        B2[i][i-2]=-af/(2*hx)

A[0:M-1,0:M-1]=B1+B2
A[M-1:2*M-2,M-1:M*2-2]=B2+B1


def g1(t,x):
    return np.exp(-((pi)**2) *bt*t)*(-gm*np.sin(pi*(x-t))+(af-1)*pi*np.cos(pi*(x-t)))+np.exp(-3*((pi)**2) *bt*t)*((np.sin(pi*(x-t)))**2)*np.cos(pi*(x-t))
def g2(t,x):
    return np.exp(-((pi)**2) *bt*t)*(-gm*np.cos(pi*(x-t))-(af-1)*pi*np.sin(pi*(x-t)))+np.exp(-3*((pi)**2 )*bt*t)*((np.cos(pi*(x-t)))**2)*np.sin(pi*(x-t))
def fun1(t,z):
    U=np.dot(A,z).reshape((2*M-2,1))
    b=np.zeros((2*M-2,1))
    u=z[0:M-1].copy()
    v=z[M-1:2*M-2].copy()
    #print(t)
    for j in range(0,M-1):
        if j==0:
           b[j]=-(u[j]**2)*v[j]+g1(t,x[j+1])+(bt/(hx**2))*(np.exp(-((pi)**2) *bt*t)*(np.sin(-pi*t)))+(af/hx)*(np.exp(-((pi)**2) *bt*t)*(np.sin(-pi*t)))
           b[M-1+j]=-(v[j]**2)*u[j]+g2(t,x[j+1])+(bt/(hx**2))*(np.exp(-((pi)**2) *bt*t)*(np.cos(-pi*t)))+(af/(hx))*(np.exp(-((pi)**2) *bt*t)*(np.cos(-pi*t)))
        if j==1:
            b[j]=-(u[j]**2)*v[j]+g1(t,x[j+1])-(af/(2*hx))*(np.exp(-((pi)**2) *bt*t)*(np.sin(-pi*t)))
            b[M-1+j]=-(v[j]**2)*u[j]+g2(t,x[j+1])-(af/(2*hx))*(np.exp(-((pi)**2) *bt*t)*(np.cos(-pi*t)))
        elif 1<j<M-2:
           b[j]=-(u[j]**2)*v[j]+g1(t,x[j+1])
           b[M-1+j]=-(v[j]**2)*u[j]+g2(t,x[j+1])
        elif j==M-2:
            b[j]=-(u[j]**2)*v[j]+g1(t,x[j+1])+(bt/(hx**2))*(np.exp(-((pi)**2) *bt*t)*(np.sin(pi*(1-t))))
            b[M-1+j]=-(v[j]**2)*u[j]+g2(t,x[j+1])+(bt/(hx**2))*(np.exp(-((pi)**2) *bt*t)*(np.cos(pi*(1-t))))


    b=b.reshape((2*M-2,1))
    return U+b
def err(x,y,tc,h):
    x1=x.reshape((2*M-2,1))
    y1=y.reshape((2*M-2,1))
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
    while fg > 1e-4*R and fg1<40:
        Rv1=Rv+e*(f1-f2)/np.linalg.norm(f1-f2)
        f1=fun1(x,Rv1)
        R=np.linalg.norm(f1-f2)/e
        fg=np.abs(R-Rr)
        fg1+=1
        Rr=R 
    if fg1==40:
        R=1.1*R
    return R,fg1
'''
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
'''


def RKC(fun1,t0,t_end,h,u0,s): 
    h1=h
    tc=[t0] #t的初始
    y=u0
    counter=0
    fg1=0
    nfe=0
    s_max=0
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
        if s==3:
         w0=1+(100)/((s)**2)
        if s==4:
         w0=1+(4)/((s)**2)
        if s==5:
         w0=1+(2)/((s)**2)
        if s==6:
         w0=1+(1.5)/((s)**2)
        if s==7:
         w0=1+(1.5)/((s)**2)
        if s==8:
         w0=1+(1.0)/((s)**2)
        if s>=9:
         w0=1+(0.9)/((s)**2)
        cc=t3(w0)*t5(w0)/(t4(w0)**2)
        yt=1/np.sqrt(cc)
        if counter==1:
            h=yt*h1
        c=np.zeros(s+1)
        b=np.zeros(s+1)
        t=np.zeros(s+1)
        t1=np.zeros(s+1)
        k0=np.zeros((2*M-2,1))
        k1=np.zeros((2*M-2,1))
        k2=np.zeros((2*M-2,1))
        k3=np.zeros((2*M-2,1))
        ky0=np.zeros((2*M-2,1))
        ky1=np.zeros((2*M-2,1))
       
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
        k0=k0.reshape((2*M-2,1))
        ky0=fun1(tc[-1],k0)
        k1=k0+u1[1] *h *ky0
        ky1=fun1(tc[-1]+u1[1]*h,k1)
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
            v1[j]=-(1-b[j-1]*t[j-1])*u1[j]
            c[j]=u[j]*c[j-1]+v[j]*c[j-2]+u1[j]+v1[j]
            k3=u[j]*k2+v[j]*k1+(1-u[j]-v[j])*k0+u1[j]*h*ky1+v1[j]*h*ky0
            #if j==4:
                #print(k[4])
            ky1=fun1(tc[-1]+c[j]*h,k3)
            k1=k2.copy()
            k2=k3.copy()
       
     
        r=1
        
        #yt=0.8
        bn=(1+r)/(yt*(1+r*yt))
        bf1=(r**2)*(1-yt)/(1+yt*r)
        b0=1-bn-bf1
        C=1/6+bf1/6-bn*(yt**3)*cc/6
        
        if counter==0:
            yc=k3.copy()
           # print(yc
           # fac=0.8*((1/err1)**(1/3))
            y = np.column_stack((y, yc))
            counter=1
            tc.append(tc[-1]+h1)
            pu,fg1=ro(tc[-1]+h1,yc)
            s2=math.sqrt(h1*pu/0.4)
            s=math.ceil(s2)
            if s_max<s:
                   s_max=s
            if s<3:
                    s=3
            if s>200:
                s=200
           
            
         
            


           
        else :
            k02=y[:,-2].copy()
            k02=k02.reshape((2*M-2,1))
            yb=k3.copy()
            yc=bf1*k02+b0*k0+bn*yb
            pu,fg1=ro(tc[-1]+h1,yc)
            tc.append(tc[-1]+h1)
            if tc[-1] + h1 > t_end:
                 h1 = t_end -tc[-1]
            s2=np.sqrt(h1*pu/0.4)                                           
            s=math.ceil(s2)
            if s<3:
                s=3
            if s>s_max:
                s_max=s
            if s>200:
                s=200  
            #h=h1
            y = np.column_stack((y, yc))

    return np.array(tc),np.array(y),nfe,s_max
t0=0
t_end=1
h=0.002
eig3,fg1=ro(0,y)
print('eig:',eig3)
eig1,abcd=np.linalg.eig(A)
eig2=np.max(np.abs(eig1))
print('eig2:',eig2)
s2=np.sqrt(h*eig3/0.4)                                           
s=int(s2)
f=fun1(0,y)
#print(y)
if s<=3:
    s=3
tc,y,nfe,s_max=RKC(fun1,t0,t_end,h,y,s)
#err=sum([(x - y) ** 2 for x, y in zip(y[1:M,-1], solu[1:M])] )/ len(solu[1:M])
err=sum([(x - y) ** 2 for x, y in zip(y[0:2*M-2,-1], solu[0:2*M-2])] )/ len(solu[0:2*M-2])
print("twostepRKC")
print("h:",h)
print("bt:",bt)
print("af:",af)
print("gm:",gm)
print("err:",np.sqrt(err))
print("nfe:",nfe)
plt.plot(x[0:M-1], y[0:M-1,-1],'red')
plt.plot(x[0:M-1], solu[0:M-1],'blue')
plt.title(' t=2 af=0.1 beta=0.05  numberical solutions of RKC')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()
