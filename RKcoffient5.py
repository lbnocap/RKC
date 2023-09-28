import numpy as np   #改造真三阶二步非线性系数
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev
import time
import math
np.seterr(divide='ignore', invalid='ignore')

us1= np.zeros((251, 1), dtype=object)
bs=  np.zeros((251, 1), dtype=object)
us=  np.zeros((251, 1), dtype=object)
vs=  np.zeros((251, 1), dtype=object)
vs1= np.zeros((251, 1), dtype=object)
cs=  np.zeros((251, 1), dtype=object)
xxs= np.zeros((251, 1), dtype=object)
As=  np.zeros((251, 1), dtype=object)
for i in range(0,3):
    us1[i,0]=1
    us[i,0]=1
    vs1[i,0]=1
    vs[i,0]=1
    cs[i,0]=1
    bs[i,0]=1
    As[i,0]=1
    xxs[i,0]=1




for s in range(3,251):
        cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
        cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
        t3=cheb_poly.deriv(1)
        t4=cheb_poly.deriv(2)
        t5=cheb_poly.deriv(3)
        t42=chebyshev.Chebyshev([0] * (2 + 1))
        s1=np.ceil(s/2)
        s1=int(s1)
        t42.coef[-1]=1
        t22=t42.deriv(2) 
        w0=1+(0.05)/((s)**2)
        w0=1+(0.7)/((s)**2)
        AA=np.zeros((s+1,s+1))
        c=np.zeros(s+1)
        b=np.zeros(s+1)
        t=np.zeros(s+1)
        t1=np.zeros(s+1)
        u1=np.zeros(s+1)
        x=np.zeros(s+1)
        e1=np.ones((s+1,1))
        x[0],x[1]=0,0 
        c[0]=0
        b[0]=1
        t[0]=1
        t[1]=w0
        w1=1/(0.25*(s**2))
        t1[0]=0
        t1[1]=1
        b[1]=1/t[1]
        u=np.zeros(s+1)
        u[0],u[1]=0,0
        v=np.zeros(s+1)
        v[0],v[1]=0,0  
        u[0],u1[1]=0,w1/w0
        c[1]=u1[1]
        AA[1,0]=u1[1]
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
         AA[j,:]=u[j]*AA[j-1,:]+v[j]*AA[j-2,:] 
         AA[j,j-1]=u1[j]  
         AA[j,0]=AA[j,0]

        c[1]=u1[1]
        a=AA[s,:].copy()
        a1=np.dot(a,e1)
        a2=np.dot(a,np.dot(AA,e1))
        a3=np.dot(a,np.dot(AA,np.dot(AA,e1)))
        a4=np.dot(a,np.dot(AA,e1)**2)/2
        b1,b2,b3,b4=-1+a1[0],1/2-a1[0]+a2[0],-1/6+a1[0]/2-a2[0]+a3[0],-1/6+a1[0]/2+a4[0]-a2[0]
        c=AA[s1,:].copy()
        c1=np.dot(c,e1)
        c2=np.dot(c,np.dot(AA,e1))
        c4=np.dot(c,np.dot(AA,e1)**2)/2
        c3=np.dot(c,np.dot(AA,np.dot(AA,e1)))
        coefficients = np.array([[1,1,1,1,1],
                         [-1, b1,0,a1[0],c1[0]],
                        [1/2, b2,0,a2[0],c2[0]],
                        [-1/6, b3,0,a3[0],c3[0]],
                        [-1/6  , b4,0,a4[0],c4[0]]])
        constants = np.array([1,1, 1/2, 1/6,1/6])
        xx=np.linalg.solve(coefficients, constants)
        xx1=xx[0]
        xx2=xx[1]
        xx3=xx[2]
        xx4=xx[3]
        xx5=xx[4]
        xx=xx.reshape((5,1))
        us1[s,0]=u1
        us[s,0]=u
        vs[s,0]=v
        cs[s,0]=c
        bs[s,0]=b
        As[s,0]=AA
        xxs[s,0]=xx

np.savez('ture3rdRKC.npz', us1=us1,vs=vs,us=us,bs=bs,As=As,xxs=xxs,cs=cs)



