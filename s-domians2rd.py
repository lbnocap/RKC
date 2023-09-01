import numpy as np
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt
import numpy as np
from sympy import *


# 假设s为切比雪夫多项式的阶数
s =10
# 创建s阶切比雪夫多项式对象
s1=np.ceil(s/2)
s1=int(s1)
s2=s1


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
c=np.zeros(s+1)
b=np.zeros(s+1)
t=np.zeros(s+1)
x=np.zeros(s+1)
k=np.zeros(s+1)
w0=1+(0.9/((s)**2))
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
w11=1/(0.31*(s**2))
u1[0],u1[1]=0,w11/w0
c[1]=w11/w0
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
        
k[s1]=t5s1(w0)*(w11**3)/(cheb_polys1(w0)*6)
k[s2]=t5s2(w0)*(w11**3)/(cheb_polys2(w0)*6) 
k[s]=t5(w0)*(w11**3)/(cheb_poly(w0)*6)  
coefficients = np.array([[c[s1], c[s]],
                         [x[s1], x[s]],
                        ])

constants = np.array([1, 1/2])
xx=np.linalg.solve(coefficients, constants)
x1,x2=xx[0],xx[1]
print(t4s1(w0)*(w11**2)/cheb_polys1(w0),x[s1]*2)
print(x1*x[s1]+x2*x[s])
def complex_function(z):
    return cheb_poly(z)/cheb_poly(w0)
w1=cheb_poly(w0)/t3(w0)
bb=cheb_poly(w0)
bs=bb/(t3(w0)*w11) 
r=1
cc=bs*(t5(w0)/bb)*(w11**3)
c=x1*t5s1(w0)*(w11**3)/cheb_polys1(w0)+x2*t5(w0)*(w11**3)/cheb_poly(w0)
#yt=((r**3+bf1)/(cc*bn*(r**3)))**(1/3)
#yt=(r-1+np.sqrt((1-r)**2+4*r))/(2*r)
#print(yt)
yt=1/np.sqrt(c)
#yt=0.6
xs=Symbol("xs")
expr=(1+r)*cc*(r**3)*(xs**3)/(xs+2*x[s]*r*(xs**2))-r*(1+r)/(1+2*x[s]*bs*r*xs)-r**3
#bn=(1+r)/(bs*(yt*c[s]+2*x[s]*r*(yt**2)))
#bf1=(c[s]*r*(1+r))/(c[s]+2*x[s]*r*yt)-r
bn=(1+r)/(yt*(1+yt*r))
bf1=(r**2)*(1-yt)/(1+yt*r)
b0=1-bf1-bn
x = np.linspace(-140, 0, 1000)
y = np.linspace(-40, 40, 1000)
X, Y = np.meshgrid(x, y)
Z =w0+w11*( X + 1j*Y)
z1=w0+w1*( X + 1j*Y)
zn=w0+w11*yt*(X+1j*Y)
values = 1-bs+bs*cheb_poly(Z)/cheb_poly(w0)
values1 = cheb_poly(z1)/cheb_poly(w0)
valuesn=(1-bs+bs*cheb_poly(zn)/cheb_poly(w0))
threerdvalues=(1-x1-x2)+x1*cheb_polys1(zn/cheb_polys1(w0))+x2*cheb_poly(zn)/cheb_poly(w0)
result=(b0+bn*(valuesn)-np.sqrt((b0+bn *(valuesn))**2+4*bf1))/2 
result1=(b0+bn*(valuesn)+np.sqrt((b0+bn *(valuesn))**2+4*bf1))/2 
result3=(b0+bn*(threerdvalues)-np.sqrt((b0+bn *(threerdvalues))**2+4*bf1))/2 
result13=(b0+bn*(threerdvalues)+np.sqrt((b0+bn *(threerdvalues))**2+4*bf1))/2
#max_real = np.max(np.real(values1[real_mask]))
plt.figure(figsize=(8, 6))
#contour=plt.contour(X, Y, np.abs(values), levels=[1], colors='red')  # 使用绝对值表示等高线高度，设置等高线值为1，颜色为红色
#plt.contour(X, Y, np.abs(threerdvalues), levels=[1], colors='blue')
contour=plt.contour(X, Y, np.abs(result3), levels=[1], colors='red')
plt.contour(X, Y, np.abs(result13), levels=[1], colors='blue')
#leftmost_contour = contour.collections[0]
#leftmost_point = leftmost_contour.get_paths()[0].vertices[0]
#leftmost_value = complex_function(leftmost_point[0])
mask1 = np.abs(result13) <= 1
mask = np.abs(result3) <=1
C=1/6+bf1/6-bn*(bs*t5(w0)*(w1**3)*(yt**3)/(6*bb))
plt.imshow(mask,extent=[-140,0,-40,40] ,origin='lower', cmap='Blues', alpha=0.5)
plt.imshow(mask1,extent=[-140,0,-40,40] ,origin='lower', cmap='Blues', alpha=0.5)
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title('Contour Plot of Complex Function (|root| = 1) s=30')
plt.show()  