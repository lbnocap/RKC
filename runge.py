import numpy as np
import matplotlib.pyplot as plt
import math 
plt.rcParams['font.sans-serif']=['SimHei']


M=600
x0=-math.pi
x_end=math.pi
x=np.linspace(x0,x_end,M+1,dtype=float)
hx=x[1]-x[0]
A=np.zeros((M+1,M+1))
y=np.zeros((M+1,1))
bt=0.05
af=0.1
A=np.zeros((M+1,M+1))
A[0][0],A[0][1],A[0][2],A[0][3]=2*bt/(hx**2)+af/hx,-5*bt/(hx**2)-af/hx,4*bt/(hx**2),-bt/(hx**2)
A[1][0],A[1][1],A[1][2]=bt/(hx**2)+af/hx,-2*bt/(hx**2)-af/hx,bt/(hx**2)
for i in range(M+1):
    y[i]=math.sin(x[i])
    if i>=2 and i<=M-1:
        A[i][i-2],A[i][i-1]=-af/2*hx,bt/(hx**2)+4*af/2*hx
        A[i][i],A[i][i+1]=-2*bt/(hx**2)-3*af/2*hx,bt/(hx**2)
    if i==M:
        A[M][M-3],A[M][M-2]=-bt/(hx**2),4*bt/(hx**2)-af/hx
        A[M][M-1],A[M][M]=-5*bt/(hx**2)+4*af/hx,2*bt/(hx**2)-3*af/hx


def f(x, y):
    return np.dot(A,y)

def runge_kutta_4th_order(f, a, b, h, y0):
    n = int((b - a) / h)
    x = np.linspace(a, b, M + 1)
    y = np.zeros((M+1,n + 1))
    y[:,0] = y0

    for i in range(n):
        k1 = h * f(x[i], y[:,i])
        k2 = h * f(x[i] + h/2, y[:,i] + k1/2)
        k3 = h * f(x[i] + h/2, y[:,i] + k2/2)
        k4 = h * f(x[i] + h, y[:,i] + k3)

        y[:,i+1] = y[:,i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x, y

a = 0
b = 2
h = 0.1
y0 = y[:,0]

tc, y = runge_kutta_4th_order(f, a, b, h, y0)


plt.plot(x, y[:,-1], label='数值解')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

