import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import math
np.seterr(divide='ignore', invalid='ignore')
 

def fun1(x,y,yy):
    
   
    
    #return  yy,((4*v2)/y**2-1-2*(yy**2))/y
    return yy,(-0.1*yy+0.05*math.exp(-0.05*2)*math.sin(x-0.1*2)+0.1*math.exp(-0.05*2)*math.cos(x-0.1*2))/-0.05

    #return yy,-1/2*(y+(y**2-1)*yy)
    return yy,(5-yy*math.exp(y)+y**2)/1
def runge44(f,bb,b,h,y0,yy0,s,s1):
    n=int((b-bb)/h)
    x=np.linspace(bb,b,n+1)
    y=np.zeros(n+1)
    yy=np.zeros(n+1)
    y[0]=y0
    yy[0]=yy0
    w0=1+0.05/((s+s1)**2)
    w1=1
    err=np.zeros(s+s1+1) 
    b=np.zeros(s+s1+1)
    t=np.zeros(s+1+s1)
    t1=np.zeros(s+1+s1)
    c=np.zeros(s+s1+1)
    c1=np.zeros(s+s1+1)
    c1[0]=0
    #err[0]=0
    #tol=0.00000001
    t1[0]=0
    t1[1]=1
    b[0]=1
    t[0]=1
    t[1]=w0
    b[1]=1/t[1]
    c[0]=0
    k=np.zeros(s+1+s1)
    k1=np.zeros(s+s1+1)
    k2=np.zeros(s+1+s1)
    
    ky=np.zeros(s+1+s1)
    kyy=np.zeros(s+1+s1)
    k1[0]=y0
   
    ky[0]=yy0
    kyy[0]=0
    u=np.zeros(s+1+s1)
    u[0],u[1]=0,0
    v=np.zeros(s+1+s1)
    v[0],v[1]=0,0  
    u1=np.zeros(s+1+s1)
    u1[0],u1[1]=0,w1/w0  
    for j in range(2,s+1+s1):
        t[j]=2*w0*t[j-1]-t[j-2]
        t1[j]=2*t[j-1]+2*w0*t1[j-1]-t1[j-2]
        b[j]=1/t[j]  
    #print(b)  
    w1=t[s+s1]/t1[s+s1]
    
    for i in range(n):
        #if j==0:
        k[0],k2[0]=y[i],yy[i]
        #print(k[0])
                
        #if j==1:
        ky[1],kyy[1]=fun1(x[i],y[i],yy[i])
        #err[i+1]=(kyy[1]/tol)**2
        #print(ky[1])
        k[1]=y[i]+(w1/w0) *h *ky[1]
        c[1]=w1/w0
        c1[1]=1/(s1+1)*c[1]
        k1[1]=1/(s1+1)*k[1]+(1-1/(s1+1))*k1[0]
        #print(k[1])
        #print(k1[1])
        k2[1]=yy[i]+(w1/w0) *h *kyy[1]
       
            #print(k[1])
        for j in range(2,s+s1+1):
            '''
            if j==0:
                k[j],k2[j]=y[i],yy[i]
               # print(k[0])
                
            if j==1:
                ky[j],kyy[j]=fun1(x[i],y[i],yy[i])
                #print(ky[1])
                k[j]=y[i]+(w1/w0) *h *ky[j]
                k2[j]=yy[i]+(w1/w0) *h *kyy[j]
                #print(k[1])
            '''
            u[j]=2*w0*b[j]/b[j-1]
            #print(u[j])
            v[j]=-b[j]/b[j-2]
            #print(v[j])
            u1[j]=2*w1*b[j]/b[j-1]
            c[j]=u[j]*c[j-1]+v[j]*c[j-2]+u1[j]
            c1[j]=1/(s1+1)*c[j]+(1-1/(s1+1))*c1[j-1]
            k[j]=u[j]*k[j-1]+v[j]*k[j-2]+(1-u[j]-v[j])*k[0]+u1[j]*h*ky[j-1]
            #if j==4:
               # print(k[4])
            k1[j]=1/(s1+1)*k[j]+(1-1/(s1+1))*k1[j-1]
            k2[j]=u[j]*k2[j-1]+v[j]*k2[j-2]+(1-u[j]-v[j])*yy[i]+u1[j]*h*kyy[j-1]
          
            ky[j],kyy[j]=fun1(x[i]+c[j]*h,k[j],k2[j])
            #err[i+1]+=(kyy[j]/tol)**2
        #err[i+1]=(err[i+1]/s)**(1/2)
        #print(k1[s+s1-1])
        x1=(1-c1[s+s1])/(c1[s+s1-1]-c1[s+s1])
        x2=1-x1
        #print(x1)
        #print(x2)
        y[i+1]=x1*k1[s+s1-1]+x2*k1[s+s1]
        #y[i+1]=k1[s+s1]
        yy[i+1]=k2[s+s1]
    #print(t)
    #print(w0)
    return x,y,yy
'''
bb=0
b=5
h=0.1
y0=1
yy0=-10
'''
bb=-np.pi
b=np.pi
h=0.1
y0=math.exp(-0.05*2)*math.sin(bb-0.1*2)
yy0=math.exp(-0.05*2)*math.cos(bb-0.1*2)
x,y,yy=runge44(fun1,bb,b,h,y0,yy0,50,2)
plt.plot(x, y, label='Numerical Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
#print(y)
#print(yy)
#print(x)

