import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']

#定义边界函数
def ex(x):
  return np.exp(x)

def left_b(t):
    return np.exp(t)

def right_b(t):
    return np.exp(1+t)

l=1
T=1
t=0.00005#时间步长
h=0.05#空间步长
M=int(T/t)
N=int(l/h)
a=1#热传导系数
r=a*t/(h**2)#网格比
print(r)
#网格划分
x=np.linspace(0, 1,N+1,dtype=float) 
y=np.linspace(0,1,M+1,dtype=float)
U=np.zeros((M+1,N+1))
A=np.zeros((N-1,N-1))
B=np.eye(N+1)
b1=np.matlib.zeros((N-1,1))
#组装A矩阵
for i in range(N-2):
    A[i][i]=4-4*r
    A[i][i+1]=2*r
    A[i+1][i]=2*r
    A[i+1][i+1]=4-4*r
B=-(h**2)*B
for i in range(0,N+1):
    #利用边界条件和理查森外推求出u0,u1初值
     U[0][i]=ex(i*h)
     if i<=N-1:
      B[i][i+1]=h
      B[i+1][i]=h
     U[1][i]=U[0][i]+(3/2)*h*ex(i*h)
    #处理左右边界


#U[1,0:N+1]=U[0,0:N+1]+(t)*(np.dot(B, U[0,0:N+1].T))
print(A)
#处理左右边界
for i in range(M+1):
    U[i][0]=left_b(i*t)
    U[i] [N]=right_b(i*t)
#处理右端列向量b
def b(u,k):
    for i in range(N-1):
        if i==0:
         b1[i]=(2*r*u[k][0]-u[k-1][1])
        elif i==N-2:
            b1[i]=(2*r*u[k][i+2]-u[k-1][i+1])
        else:
            b1[i]=-u[k-1][i+1]
    return b1
#迭代求数值解
for i in range(2,M+1): 
    Ut=U[i-1,1:N].reshape(N-1,1)
   # if i==6:
      #print(b(U,i-1))
      #print((np.dot(A, Ut))/3+b(U,i-1)/3)
    bb=(np.dot(A, Ut))+b(U,i-1)
    bb=bb.reshape(1,N-1)
    U[i,1:N]=bb/3
#for j in range(11):
    #print(U[j])
print(U)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y,U, rstride=1, cstride=1, cmap='hot')
df = pd.DataFrame(U.T, 
                  index=x,#DataFrame的行标签设置为大写字母
                  columns=y)#设置DataFrame的列标签
plt.figure(dpi=120)
plt.title('热传导方程数值解3D图（22数学刘波）')
sns.heatmap(data=df,#矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签''' 
) 
plt.show()