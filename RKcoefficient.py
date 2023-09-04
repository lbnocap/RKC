import numpy as np
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt
#原始二阶



def RKcoefficient(e1,s):
    cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
    cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
    t3=cheb_poly.deriv(1)
    t4=cheb_poly.deriv(2)
    t5=cheb_poly.deriv(3)
    t42=chebyshev.Chebyshev([0] * (2 + 1))
    t42.coef[-1]=1
    t22=t42.deriv(2) 
    w0=1+(0.05)/((s)**2)
    c=np.zeros(s+1)
    b=np.zeros(s+1)
    t=np.zeros(s+1) 
    A=np.zeros((s+1,s+1))
    t1=np.zeros(s+1)
    x=np.zeros(s+1)   
    e=np.ones((s+1,1))
    w1=t3(w0)/t4(w0)

    x[0],x[1]=0,0 
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
    u[0],u1[1]=0,b[1]*w1      
    c[1]=u1[1]
    v[1]=-b[1]/b[0]
    v1[1]=-(1-b[0]*t[0])*u1[1                               ]
    A[1,0]=u1[1]
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
            x[j]=u[j]*x[j-1]+v[j]*x[j-2]+u1[j]*c[j-1]
            A[j,:]=u[j]*A[j-1,:]+v[j]*A[j-2,:]
            A[j,j-1]=u1[j]  
            A[j,0]=A[j,0]+v1[j]
    return A