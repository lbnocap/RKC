import numpy as np
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt
import time

# 假设s为切比雪夫多项式的阶数
s = 10
# 创建s阶切比雪夫多项式对象
cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
t3=cheb_poly.deriv(1)
t4=cheb_poly.deriv(2)
t5=cheb_poly.deriv(3)
c=np.zeros(s+1)
b=np.zeros(s+1)
t=np.zeros(s+1)
x=np.zeros(s+1)
w0=1+(2/13/((s)**2))
x[0],x[1]=0,0 
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
w1=t3(w0)/t4(w0)
#w1=t3(w0)/t4(w0)
u1[0],u1[1]=0,w1/w0
c[1]=w1/w0
for j in range(2,s+1):
         t[j]=2*w0*t[j-1]-t[j-2]
         b[j]=1/t[j] 
         u[j]=2*w0*b[j]/b[j-1]
            #print(u[j])
         v[j]=-b[j]/b[j-2]
            #print(v[j])
         u1[j]=2*w1*b[j]/b[j-1]
         c[j]=u[j]*c[j-1]+v[j]*c[j-2]+u1[j]
         x[j]=u[j]*x[j-1]+v[j]*x[j-2]+u1[j]*c[j-1]
def complex_function(z):
    return cheb_poly(z)/cheb_poly(w0)
w1=t3(w0)/t4(w0)
bb=cheb_poly(w0)
bs=t4(w0)/(t3(w0)**2)
r=1.1
cc=t3(w0)*t5(w0)/(t4(w0)**2)
#yt=((r**3+bf1)/(cc*bn*(r**3)))**(1/3)
yt=0.6
#yt=1/np.sqrt(cc)
print(yt)
bn=(1+r)/(yt*(1+yt*r))
bf1=(r**2)*(1-yt)/(1+yt*r)
b0=1-bf1-bn
x = np.linspace(-120, 0, 1000)
y = np.linspace(-30, 30, 1000)
X, Y = np.meshgrid(x, y)
Z =w0+w1*yt*( X + 1j*Y)
values = 1-bs*bb+bs*cheb_poly(Z)
result=(b0+bn*(values)-np.sqrt((b0+bn *(values))**2+4*bf1))/2 
result1=(b0+bn*(values)+np.sqrt((b0+bn *(values))**2+4*bf1))/2
plt.figure(figsize=(8, 6))
#plt.contour(X, Y, np.abs(result), levels=[1], colors='red') 
#plt.contour(X, Y, np.abs(result1), levels=[1], colors='blue') 
mask1 = np.abs(result1) <= 1
mask = np.abs(result) <=1
plt.imshow(mask,extent=[-120,0,-30,30] ,origin='lower', cmap='Blues', alpha=0.5)
plt.imshow(mask1,extent=[-120,0,-30,30] ,origin='lower', cmap='Blues', alpha=0.5)
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title('Contour Plot of Complex Function (|root| = 1)')
plt.show()