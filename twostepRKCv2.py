import numpy as np  #改造二阶二步
import numpy.matlib
import matplotlib.pyplot as plt
import math
from numpy.polynomial import chebyshev
import time
np.seterr(divide='ignore', invalid='ignore') 
M=600
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
tol=1e-4
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
    return (12*(x-y)+6*h*(np.dot(A,x)+np.dot(A,y)))
def fun1(x,y):
     return np.dot(A,y)
def RKC(f,t0,t_end,h,u0,s):
    h_v=[h]
    h1=h
    tc=[t0] #t的初始
    y=u0
    s_max=0
    nfr=0
    counter=0
    while tc[-1]<t_end:
        nfr+=s 
        cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
        cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
        t3=cheb_poly.deriv(1)
        t4=cheb_poly.deriv(2)
        t5=cheb_poly.deriv(3)
        w0=1+(0.05)/((s)**2)
        c=np.zeros(s+1)
        b=np.zeros(s+1)
        t=np.zeros(s+1)
        t1=np.zeros(s+1)
        k=np.zeros((M+1,4))
        ky=np.zeros((M+1,2))
        u1=np.zeros(s+1)
        x=np.zeros(s+1)
        x[0],x[1]=0,0 
        c[0]=0
        b[0]=1
        t[0]=1
        t[1]=w0
        w1=1/(0.341*(s**2))
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
        k[:,0]=y[:,-1]
        ky[:,0]=fun1(t[-1],y[:,-1])
        k[:,1]=y[:,-1]+u1[1] *h *ky[:,0]
        ky[:,1]=fun1(t[-1]+u1[1]*h,k[:,1])
        k[:,2]=k[:,1]
        k[:,1]=k[:,0]
        for j in range(2,s+1):
            cheb_poly1 = chebyshev.Chebyshev([0] * (j + 1))
            cheb_poly1.coef[-1] = 1
            tj=cheb_poly1.deriv(2)
            k[:,3]=u[j]*k[:,2]+v[j]*k[:,1]+(1-u[j]-v[j])*k[:,0]+u1[j]*h*ky[:,1]
            #if j==4:
                #print(k[4])
            ky[:,1]=fun1(tc[-1]+c[j]*h,k[:,3])
            k[:,1]=k[:,2]
            k[:,2]=k[:,3]
        if counter==0:
            r=1
        if counter>=1:
            r=h1/tc[-1]
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
            yc=(1-bs)*y[:,-1]+bs*k[:,3]
            err2=C*err(y[:,-1],yc,h1)/1e-2
            err1=np.linalg.norm(err2)/math.sqrt(M+1)
            fac=0.8*((1/err1)**(1/3))
            print(C)
            if err1<1 or h1==0.0005:
                y = np.column_stack((y, yc))
                counter+=1
                h=yt*h1
                tc.append(tc[-1]+h1)
                s_max=s
            if err1>1:
                h1=max(fac,0.1)*h1
                if h1<0.0005:
                    h1=0.0005
                h=h1
                s2=math.sqrt(h1*eig2/1.138)
                s=math.ceil(s2)
                if s<3:
                    s=3
        else :
            yb=(1-bs)*y[:,-1]+bs*k[:,3]
            yc=bf1*y[:,-2]+b0*y[:,-1]+bn*yb
            err2=C*err(y[:,-1],yc,h1)/tol
            err1=np.linalg.norm(err2)/math.sqrt(M+1)
            fac=0.8*((1/err1)**(1/3))
            if err1<1 or h1==0.0005:
               y = np.column_stack((y, yc))
               if tc[-1] + h1 > t_end:
                 h1 = t_end -tc[-1]
                 h=yt*h
               tc.append(tc[-1]+h1)
               if s>s_max:
                    s_max=s
            if err1>1:
                h1=max(fac,0.1)*h1
                if h1<0.0005:
                    h1=0.0005
                h=yt*h1
                s2=math.sqrt(h1*eig2/1.138)
                s=math.ceil(s2)
                if s<3:
                    s=3
    return np.array(tc),np.array(y),s_max,nfr
t0=0
t_end=2
h=0.01
eig1,abcd=np.linalg.eig(A)
eig2=np.max(np.abs(eig1))
s2=math.sqrt(h*eig2/1.138)
s=math.ceil(s2)
if s<=3:
    s=3
tc,y,s_max,nfr=RKC(fun1,t0,t_end,h,y,s)
#mse = np.mean((np.array(y[1:M,-1]) - np.array(solu[1:M]))**2)
#mae = np.mean(np.abs(np.array(y[1:M,-1]) - np.array(solu[1:M])
# ))
err=sum([(x - y) ** 2 for x, y in zip(y[1:M,-1], solu[1:M])] )/ len(solu[1:M])
print(np.sqrt(err))
time_end=time.time()
print(time_end-time_st)
print(len(tc))
print(s_max)
print("评估次数：",nfr)
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