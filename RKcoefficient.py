import numpy as np
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt
import h5py

# 假设s为切比雪夫多项式阶数
us1= np.zeros((251, 1), dtype=object)
bs=  np.zeros((251, 1), dtype=object)
us=  np.zeros((251, 1), dtype=object)
vs=  np.zeros((251, 1), dtype=object)
vs1= np.zeros((251, 1), dtype=object)
cs=  np.zeros((251, 1), dtype=object)
xxs= np.zeros((251, 1), dtype=object)
As=  np.zeros((251, 1), dtype=object)
for i in range(0,5):
  us1[i,0]=1
  us[i,0]=1
  vs1[i,0]=1
  vs[i,0]=1
  cs[i,0]=1
  bs[i,0]=1
  As[i,0]=1
  xxs[i,0]=1
       

for s in range(5,52):
# 创建s阶切比雪夫多项式对象
   cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
   cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
   t3=cheb_poly.deriv(1)
   t4=cheb_poly.deriv(2)
   t5=cheb_poly.deriv(3)
   t42=chebyshev.Chebyshev([0] * (2 + 1))
   t42.coef[-1]=1
   t22=t42.deriv(2) 
   w0=1+(4/((s)**2))
   c=np.zeros(s+1)
   b=np.zeros(s+1)
   t=np.zeros(s+1) 
   A=np.zeros((s+1,s+1))
   t1=np.zeros(s+1)
   x=np.zeros(s+1)   
   e=np.ones((s+1,1))
   #w1=t3(w0)/t4(w0)
   w1=(1+w0)/(0.45*(s**2))
   #w1=1/(0.22* s**2)
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
   v1[1]=-(1-b[0]*t[0])*u1[1]
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
             v1[j]=-(1-b[j-1]*t[j-1])*u1[j]
             c[j]=u[j]*c[j-1]+v[j]*c[j-2]+u1[j]+v1[j]
             x[j]=u[j]*x[j-1]+v[j]*x[j-2]+u1[j]*c[j-1]
             A[j,:]=u[j]*A[j-1,:]+v[j]*A[j-2,:]
             A[j,j-1]=u1[j]  
             A[j,0]=A[j,0]+v1[j]
   a=A[s,:].copy()
   a1=np.dot(a,e)
   a2=np.dot(a,np.dot(A,e))
   a3=np.dot(a,np.dot(A,np.dot(A,e)))
   a4=np.dot(a,np.dot(A,e)**2)/2
   b1,b2,b3,b4=-1 +a1[0],1/2-a1[0]+a2[0],-1/6+a1[0]/2-a2[0]+a3[0],-1/6+a1[0]/2+a4[0]-a2[0]
   coefficients = np.array([[1,1,1,1],
                         [-1, b1,0,a1[0]],
                        [1/2, b2,0,a2[0]],
                        [-1/6, b3,0,a3[0]]])

   constants = np.array([1,1, 1/2, 1/6])
   xx=np.linalg.solve(coefficients, constants)
   xx1=xx[0]
   xx2=xx[1]
   xx3=xx[2]
   xx4=xx[3]
   xx=xx.reshape((4,1))
   us1[s,0]=u1
   us[s,0]=u
   vs1[s,0]=v1
   vs[s,0]=v
   cs[s,0]=c
   bs[s,0]=b
   As[s,0]=A
   xxs[s,0]=xx
print(us1.dtype)

'''
with h5py.File('widetwostepRKCv2153.h5', 'a') as hf:
    # 保存多个数组，每个数组都可以在文件中表示为一个数据集
    hf.create_dataset('us1', data=np.array(us1,dtype=object))
    
    hf.create_dataset('u_', data=us)
    hf.create_dataset('v_1', data=vs1)
    hf.create_dataset('v_', data=vs)
    hf.create_dataset('b_', data=bs)
    hf.create_dataset('As_', data=As)
    hf.create_dataset('xxs_', data=xxs)
    hf.create_dataset('c_', data=cs)'''


