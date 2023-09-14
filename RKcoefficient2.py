import numpy as np
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
#一阶RKC系数

def  RKcoefficient2(e1,s,www):
    s1=np.ceil(s/3)
    s1=int(s1)
    s2=2*s1
    cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
    cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
    cheb_polys1 = chebyshev.Chebyshev([0] * (s1 + 1))
    cheb_polys1.coef[-1] = 1
    cheb_polys2 = chebyshev.Chebyshev([0] * (s2 + 1))
    cheb_polys2.coef[-1] = 1
    t3=cheb_poly.deriv(1)
    t4s1=cheb_polys1.deriv(2)
    t4s2=cheb_polys2.deriv(2)
    t4=cheb_poly.deriv(2)
    t5=cheb_poly.deriv(3)
    t5s1=cheb_polys1.deriv(3)
    t5s2=cheb_polys2.deriv(3)
    A=np.zeros((s+1,s+1))
    c=np.zeros(s+1)
    b=np.zeros(s+1)
    t=np.zeros(s+1)
    x=np.zeros(s+1)
    k=np.zeros(s+1)
    e=np.ones(s+1)

    w0=1+(www/((s)**2))
    x[0],x[1]=0,0 
    k[0],k[1]=0,0
    c[0]=0
    b[0]=1
    t[0]=1
    t[1]=w0
    b[1]=1/t[1]
    u=np.zeros(s+1)
    u[0],u[1]=0,0
    v=np.zeros(s+1)
    v[0],v[1]=0,0  
    u1=np.zeros(s+1)
    # 计算s阶切比雪夫多项式在特定点的值\w0=
    #w11=t3(w0)/t4(w0)
    w11=1/(e1*(s**2))
    u1[0],u1[1]=0,w11/w0
    c[1]=w11/w0
    A[1,0]=u1[1]
    for j in range(2,s+1):
         t[j]=2*w0*t[j-1]-t[j-2]
         b[j]=1/t[j] 
         u[j]=2*w0*b[j]/b[j-1]
            #print(u[j])
         v[j]=-b[j]/b[j-2]
            #print(v[j])
         u1[j]=2*w11*b[j]/b[j-1]
         c[j]=u[j]*c[j-1]+v[j]*c[j-2]+u1[j]
         x[j]=u[j]*x[j-1]+v[j]*x[j-2]+u1[j]*c[j-1]
         A[j,:]=u[j]*A[j-1,:]+v[j]*A[j-2,:]
         A[j,j-1]=u1[j]  
         A[j,0]=A[j,0]
    a=A[s,:].copy()
    a1=np.dot(a,e)
    a2=np.dot(a,np.dot(A,e))
    a4=np.dot(a,np.dot(A,e)**2)/2
    a3=np.dot(a,np.dot(A,np.dot(A,e)))
    return  -1+a1,1/2-a1+a2,-1/6+a1/2-a2+a3,-1/6+a1/2+a4-a2
