import numpy as np
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt
import time

# 假设s为切比雪夫多项式阶数
s =100
# 创建s阶切比雪夫多项式对象
cheb_poly = chebyshev.Chebyshev([0] * (s + 1))
cheb_poly.coef[-1] = 1  # 将最高阶系数设为1，得到s阶切比雪夫多项式
t3=cheb_poly.deriv(1)
t4=cheb_poly.deriv(2)
t5=cheb_poly.deriv(3)
t42=chebyshev.Chebyshev([0] * (2 + 1))
t42.coef[-1]=1
t22=t42.deriv(2) 
w0=1+((2/13)/((s)**2))
c=np.zeros(s+1)
b=np.zeros(s+1)
t=np.zeros(s+1) 
A=np.zeros((s+1,s+1))
t1=np.zeros(s+1)
x=np.zeros(s+1)   
e=np.ones((s+1,1))
w1=t3(w0)/t4(w0)
#w1=(1+w0)/(0.45*(s**2))
#w1=1/(0.22* s**2)
x[0],x[1]=0,0 
c[0]=0  
b[0]=1
t[0]=1
t[1]=w0
t1[0]=0
t1[1]=1
rr=1
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
print(b[1])
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
print(s,u,v,u1,v1)
def complex_function(z):
    return cheb_poly(z)/cheb_poly(w0)
a=A[s,:].copy()
a1=np.dot(a,e)
a2=np.dot(a,np.dot(A,e))
a3=np.dot(a,np.dot(A,np.dot(A,e)))
a4=np.dot(a,np.dot(A,e)**2)/2
print(a1,a2,a3,a4)
print(a3-1/6)
b1,b2,b3,b4=-1+a1[0],1/2-a1[0]+a2[0],-1/6+a1[0]/2-a2[0]+a3[0],-1/6+a1[0]/2+a4[0]-a2[0]
coefficients = np.array([[1,1,1,1],
                         [-1, b1,0,a1[0]*rr],
                        [1/2, b2,0,a2[0]*(rr**2)],
                        [-1/6, b3,0,a3[0]*(rr**3)]])
print('c:',coefficients)
constants = np.array([1,rr, (rr**2)/2, (rr**3)/6])
xx=np.linalg.solve(coefficients, constants)
xx1=xx[0]
xx2=xx[1]
xx3=xx[2]
xx4=xx[3]
print(c)
print("ceeor",np.abs(-xx1/6+b4*xx2+a4[0]*xx4)-1/6)
bb=cheb_poly(w0)
bs=t4(w0)/(t3(w0)**2)
r=1
cc=t3(w0)*t5(w0)/(t4(w0)**2)
#yt=((r**3+bf1)/(cc*bn*(r**3)))**(1/3)
#yt=0.6
yt=1/np.sqrt(cc)

bn=(1+r)/(yt*(1+yt*r))
bf1=(r**2)*(1-yt)/(1+yt*r)
b0=1-bf1-bn
C=1/6+bf1/6-bn*(yt**3)*cc/6

x = np.linspace(-0.7*s**2, 0, 3000)
y = np.linspace(-1.2*s, 1.2*s, 1000)
X, Y = np.meshgrid(x, y)                                            
Z =w0+w1*( X + 1j*Y)
values = 1-bs*bb+bs*cheb_poly(Z)
values3 = xx1+xx2*(1-bs*bb+bs*cheb_poly(Z))
values31 = xx3+xx4*(1-bs*bb+bs*cheb_poly(Z))
result=(b0+bn*(values)-np.sqrt((b0+bn *(values))**2+4*bf1))/2 
result1=(b0+bn*(values)+np.sqrt((b0+bn *(values))**2+4*bf1))/2
result3=(values31+np.sqrt((values31)**2+4*values3))/2
result31=(values31-np.sqrt((values31)**2+4*values3))/2
#plt.figure(figsize=(8, 6))
#plt.contour(X, Y, np.abs(result), levels=[1], colors='red') 
#plt.contour(X, Y, np.abs(result1), levels=[1], colors='blue')
#plt.contour(X, Y, np.abs(values), levels=[1], colors='red')  
mask1 = np.abs(result3) <= 1
mask = np.abs(result31) <=1
overlap_mask = mask1 & mask
plt.imshow(overlap_mask,extent=[-0.7*s**2,0,-1.2*s,1.2*s] ,origin='lower', cmap='Blues', alpha=1,aspect='auto')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
#plt.savefig('C:/Users/A204-7/Desktop/论文撰写及其模板模板/stable_domians/NTRCKs100wide.eps')
plt.show()

c1=np.dot(b,e)
a=A[s,:].copy()
a1=np.dot(a,e)
a2=np.dot(a,np.dot(A,e))
a3=np.dot(a,np.dot(A,e)**2)
a4=np.dot(a,np.dot(A,np.dot(A,e)))
print("三阶系数误差：",np.abs(-bf1/6+bn*(yt**3)*a4[0]/2-1/6))
#print(np.dot(b,c*c))
#-1.3*(1+w0)/w1