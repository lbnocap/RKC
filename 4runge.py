import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns

 

def fun1(x,y,yy):
    return yy,-1/2*(y+(y**2-1)*yy)

def runge44(f,bb,b,h,y0,yy0):
    n=int((b-bb)/h)
    x=np.linspace(bb,b,n+1)
    y=np.zeros(n+1)
    yy=np.zeros(n+1)
    y[0]=y0
    yy[0]=yy0
    
    for i in range(n):
        k1y,k1yy=fun1(x[i],y[i],yy[i])
        k2y,k2yy=fun1(x[i] + h/2,y[i]+h/2 * k1y,yy[i]+h/2 * k1yy)
        k3y,k3yy=fun1(x[i] + h/2,y[i]+h/2 * k2y,yy[i]+h/2 * k2yy)
        k4y,k4yy=fun1(x[i]+h,y[i]+h *k3y,yy[i] + h*k3yy )
        y[i+1]=y[i]+h/6 * (k1y+2*k2y+2*k3y+k4y)
        yy[i+1]=yy[i]+h/6 * (k1yy+2*k2yy+2*k3yy+k4yy)
    return x,y,yy
bb=0
b=20
h=0.1
y0=0.25
yy0=0
x,y,yy=runge44(fun1,bb,b,h,y0,yy0)
plt.plot(x, y, label='Numerical Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
