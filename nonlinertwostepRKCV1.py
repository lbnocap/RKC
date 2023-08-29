import numpy as np   #改造二阶二步非线性变步长
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import copy
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
atol=1e-4
rtol=1e-4
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

def fun1(x,z):
    b=np.zeros((3*M+3,1))
    u=z[0:M+1]
    v=z[M+1:2*M+2]
    w=z[2*M+2:3*M+3]
    
    for j in range(0,M+1):
        b[j]=(-1/bf)*(u[j]**3+u[j]*v[j])
        b[M+1+j]=0.07*(u[j]-0.7)*(u[j]-1.3)/((u[j]-0.7)*(u[j]-1.3)+0.1)
        b[2*M+2+j]=-((v[j])**2)*w[j]+0.035*(u[j]-0.7)*(u[j]-1.3)/((u[j]-0.7)*(u[j]-1.3)+0.1)
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
    v1=0.07*(u-0.7*e)*(u-1.3*e)/((u-0.7*e)*(u-1.3*e)+0.1*e)
    w1=-(v**2)*w+0.035*(u-0.7*e)*(u-1.3*e)/((u-0.7*e)*(u-1.3*e)+0.1*e)
    b=np.vstack((u1, v1, w1))'''
    b=b.reshape((603,1))
    U=np.dot(BB,z).reshape((3*M+3,1))
    return U+b
def err(x,y,h):
    x1=x.reshape((3*M+3,1))
    y1=y.reshape((3*M+3,1))
    z1=12*(x1-y1)
    return 0.2*(z1+6*h*(fun1(h,x1)+fun1(h,y1)))

eig1,abcd=np.linalg.eig(BB)
eig2=5*np.max(np.abs(eig1)) 

def err_s(x,y,y1,atol,rtol):
    ln=len(y)
    sc=y1.copy()
    sc1=y1.copy()
    for i in range(ln):
        sc[i]=(atol+rtol*max(np.abs(y[i],np.abs(y1[i]))))
        sc1[i]=x[i]/sc[i]
    sc1=sc1.reshape((ln,1))
    return np.linalg.norm(sc1)/np.sqrt(ln)
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
        R=1.2*R
    return R,fg1


def RKC(f,t0,t_end,h,u0,s):
    tc=[t0] #t的初始
    y=u0
    nfe=0
    h1=h
    h_v=[h]
    counter=0
    fg1=0
    s_max=0
    while tc[-1]<t_end:
        cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
        cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
        t3=cheb_poly.deriv(1)
        t5=cheb_poly.deriv(3)
        nfe=s+fg1+3+nfe
        w0=1+(0.1)/((s)**2)
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
        u1=np.zeros(s+1)
        x=np.zeros(s+1)
        x[0],x[1]=0,0 
        c[0]=0
        b[0]=1
        t[0]=1
        t[1]=w0
        w1=1/(0.33*(s**2))
        t1[0]=0
        t1[1]=1
        b[1]=1/t[1]
        u=np.zeros(s+1)
        u[0],u[1]=0,0
        v=np.zeros(s+1)
        v[0],v[1]=0,0  
        u[0],u1[1]=0,w1/w0
        c[1]=u1[1]
        for j in range(2,s+1):
         t[j]=2*w0*t[j-1]-t[j-2]
         t1[j]=2*t[j-1]+2*w0*t1[j-1]-t1[j-2]
         b[j]=1/t[j]
         v[j]=-b[j]/b[j-2]
         u[j]=2*w0*b[j]/b[j-1]
            #print(v[j])
         u1[j]=2*w1*b[j]/b[j-1]
         c[j]=u[j]*c[j-1]+v[j]*c[j-2]+u1[j]
         x[j]=u[j]*x[j-1]+v[j]*x[j-2]+u1[j]*c[j-1]
        k0=y[:,-1].copy()
        k0=k0.reshape((603,1))
        ky0=fun1(t[-1],k0)
        if tc[-1]==0:
            1
            #print(ky0)
            #print(bbb)
        k1=k0+u1[1] *h *ky0

        if tc[-1]==0:
            1
            #print(k1)
        ky1=fun1(t[-1]+u1[1]*h,k1)
        if tc[-1]==0:
            1
            #print(ky1)
        k2=k1.copy()
        k1=k0.copy()
        for j in range(2,s+1):
            k3=u[j]*k2+v[j]*k1+(1-u[j]-v[j])*k0+u1[j]*h*ky1

            if tc[-1]==0 and j==2:
               1
               #print(k3)
            ky1=fun1(tc[-1]+c[j]*h,k3)
            k1=k2.copy()
            k2=k3.copy()
       
        if counter==0:
            r=1
        if counter==1:
            r=h1/h_v[-1]
            print(r)
        #cc=t3(w0)*t5(w0)/(t4(w0)**2)
        bb=cheb_poly(w0)
        bs=bb/(t3(w0)*w1)
        #yt=1/np.sqrt(cc)
        yt=0.6
        bn=(1+r)/(bs*(yt*c[s]+2*x[s]*r*(yt**2)))
        bf1=(c[s]*r*(1+r))/(c[s]+2*x[s]*r*yt)-r
        b0=1-bf1-bn
        C=1/6+bf1/6-bn*(bs*t5(w0)*(w1**3)*(yt**3)/(6*bb))
        
        if counter==0:
            yc=(1-bs)*k0+bs*k3
           # print(yc)
            #err2=err(y[:,-1],yc,h1)
            #err1=err_s(err2,y[:,-1],yc,atol,rtol)
            err2=err(y[:,-1],yc,h1)/atol
            err1=np.linalg.norm(err2)/math.sqrt(M+1)
            err1=err1
            fac=0.8*((1/err1)**(1/3))
            if err1<1:
                y = np.column_stack((y, yc))
                counter+=1
                tc.append(tc[-1]+h1)
                h_v.append(h1)
                if s_max<s:
                   s_max=s
                pu,fg1=ro(tc[-1]+h1,yc)
                s2=math.sqrt(h1*pu/0.9)
                s=math.ceil(s2)
                h1=min(max(fac,0.1),1.1)*h1
                h=yt*h1


            if err1>1:
                h1=min(max(fac,0.1),1.1)*h1
                h=h1
                pu,fg1=ro(tc[-1]+h1,yc)
                s2=math.sqrt(h1*pu/0.9)
                s=math.ceil(s2)
                if s<3:
                    s=3
              
        else :
            k02=y[:,-2].copy()
            k02=k02.reshape((603,1))
            yb=(1-bs)*k0+bs*k3
            yc=bf1*k02+b0*k0+bn*yb
            err2=err(y[:,-1],yc,h1)
            #if tc[-1]+h1==t_end:
                #print(yc)
            err1=err_s(err2,y[:,-1],yc,atol,rtol)
            #err2=err(y[:,-1],yc,h1)/atol
            #err1=np.linalg.norm(err2)/math.sqrt(3*M+3)
            print(err1)
            err4=np.linalg.norm(err2)/math.sqrt(3*M+3)
            print("err:",err4)
            fac=0.8*((1/err1)**(1/3))
            pu,fg1=ro(tc[-1]+h1,yc)
            if err1<1 or h1==0.0001:
                y = np.column_stack((y, yc))
                tc.append(tc[-1]+h1)
                h_v.append(h1)
                h1=min(max(fac,0.1),1.1)*h1
                if  tc[-1] + h1 > t_end:
                    h1 = t_end -tc[-1]
                s2=np.sqrt(h1*pu/0.9) 
                s=math.ceil(s2)
                if s<3:
                    s=3
                h=yt*h1
                if s>s_max:
                    s_max=s
            else:
                h1=min(max(fac,0.1),1.1)*h1
                if h1<0.0001:
                    h1=0.0001
                if  tc[-1] + h1 > t_end:
                    h1 = t_end -tc[-1]
                s2=np.sqrt(h1*pu/0.9) 
                s=math.ceil(s2)
                h=yt*h1
                if s<3:
                    s=3
        
    return np.array(tc),np.array(y),nfe,s_max,h1
t0=0
t_end=1.1
h=0.01
d0=err_s(y,y,y,atol,rtol)
f0=fun1(0,y)
d1=err_s(f0,y,y,atol,rtol)
if d0<10**(-5):
    h0=10^(-6)
elif d1<10**(-5):
    h0=10^(-6)
else:
    h0=0.01*(d0/d1)
yh1=y+h0*f0
f1=fun1(0,yh1)
d2=err_s(f0-f1,y,y,atol,rtol)/h0
d2=max(d1,d2)
if d2<10**(-15):
    h01=max(10**(-6),h0*10**(-3))
else:
    h01=(0.01/d2)**(1/3)
h=min(100*h0,h01)
print(h)
eig3,fg1=ro(0,y)
print(eig2,fg1)
print('eig:',eig3)
s2=np.sqrt(h*eig3/0.9)                                           
s=int(s2)
#print(fun1(x,y))
#print(y)
if s<=3:
    s=3
tc,y,nfr,s_max,h1=RKC(fun1,t0,t_end,h,y,s)

#mse = np.mean((np.array(y[1:M,-1]) - np.array(solu[1:M]))**2)
#mae = np.mean(np.abs(np.array(y[1:M,-1]) - np.array(solu[1:M])
# ))
#err=sum([(x - y) ** 2 for x, y in zip(y[1:M,-1], solu[1:M])] )/ len(solu[1:M])
#rint(np.sqrt(err))
time_end=time.time()
print(tc)
print('时间：',time_end-time_st)
print("步数：",len(tc))
print("评估次数：",nfr)
print("s_max:",s_max)
err2=err(y[:,-2],y[:,-1],h1)
#print(y[:,3])
err1=np.linalg.norm(err2)/math.sqrt(3*M+3)
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

