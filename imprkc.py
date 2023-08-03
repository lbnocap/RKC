import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math 
np.seterr(divide='ignore', invalid='ignore')
 

def fun1(x,y,yy):
    return yy,(0.1*yy-0.05*math.exp(-0.05*2)*math.sin(x-0.1*2)-0.1*math.exp(-0.05*2)*math.cos(x-0.1*2))/0.05
    #return yy,(5-yy*ma th.exp(y)+y**2)/1
    #return yy,(1-y**2)*yy-y
def runge44(f,bb,b1,h,y0,yy0,s,s1):
    #x=np.linspace(bb,b,n+1)
    #y=np.zeros(n+1)
    #yy=np.zeros(n+1)
    h_v=[h]
    x=[bb]
    y=[y0]
    yy=[yy0]
    w0=1+0.05/((s+s1)**2)
    w1=1
    err=[0] 
    b=np.zeros(s+s1+1)
    t=np.zeros(s+1+s1)
    t1=np.zeros(s+1+s1)
    k1=np.zeros(s+s1+1)
    c=np.zeros(s+s1+1)
    c1=np.zeros(s+s1+1)
    c1[0]=0
    c[0]=0
    k1[0]=y0
    #err[0]=0
    tol=0.00001
    t1[0]=0
    t1[1]=1
    b[0]=1
    t[0]=1
    t[1]=w0
    b[1]=1/t[1]
    k=np.zeros(s+1+s1)
    k2=np.zeros(s+1+s1)
    ky=np.zeros(s+1+s1)
    kyy=np.zeros(s+1+s1)
    ky[0]=yy0
    kyy[0]=0
    u=np.zeros(s+1+s1)
    u[0],u[1]=0,0
    v=np.zeros(s+1+s1)
    v[0],v[1]=0,0  
    u1=np.zeros(s+1+s1)
    u1[0],u1[1]=0,w1/w0  
    for j in range(2,s+s1+1):
        t[j]=2*w0*t[j-1]-t[j-2] 
        t1[j]=2*t[j-1]+2*w0*t1[j-1]-t1[j-2]
        b[j]=1/t[j]  
    #print(b)  
    w1=t[s+s1]/t1[s+s1]

    
    while x[-1]<b1:
        h=h_v[-1]
        if x[-1] + h > b1:
            h = b1 - x[-1]
        #if j==0:
        k[0],k2[0]=y[-1],yy[-1] 
        #print(k[0])
                
        #if j==1:
        ky[1],kyy[1]=fun1(x[-1],y[-1],yy[-1])
        #err[i+1]=(kyy[1]/tol)**2
        #print(ky[1])
        k[1]=y[-1]+(w1/w0) *h *ky[1]
        c[1]=w1/w0
        c1[1]=1/(s1+1)*c[1]
        k1[1]=1/(s1+1)*k[1]+(1-1/(s1+1))*k1[0]
        k2[1]=yy[-1]+(w1/w0) *h *kyy[1] 
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
                #print(k[4])
            k1[j]=1/(s1+1)*k[j]+(1-1/(s1+1))*k1[j-1]
            k2[j]=u[j]*k2[j-1]+v[j]*k2[j-2]+(1-u[j]-v[j])*yy[-1]+u1[j]*h*kyy[j-1]
            ky[j],kyy[j]=fun1(x[-1]+c[j]*h,k[j],k2[j])
        x1=(1-c1[s+s1])/(c1[s+s1-1]-c1[s+s1])
        x2=1-x1
        #print(kyy[s+s1])
        err1=(1/2 *h* h*kyy[s+s1]/tol)**2
        err1=(err1)**(1/2)
        err.append(err1)
        h1=0.8*h*((1/err1)**(1/2))
        if h1/h<=5 and h1/h>=0.1:
            h=h1
        else:
            h=h
        h_v.append(h)
        x.append(x[-1]+h)
        y.append(x1*k1[s+s1-1]+x2*k1[s+s1])
        yy.append(k2[s+s1])
    #print(t)
    #print(w0)
    #print(h_v)
    return np.array(x),np.array(y),np.array(yy)
'''
bb=0
b=20
h=0.1
y0=2
yy0=0
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
#print(y0)
print(bb)
#print(yy)
#print(x)