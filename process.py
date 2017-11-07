import numpy as np
import math
import matplotlib.pyplot as plt

#calculate ax*b
def lineFun(a,b,X):
    return [a*x+b for x in X]
def expFun(a,b,X):
    return [b*math.exp(a*t) for t in X]
#transform domain from [ymin,ymax] to [-S,S]
def transform(N,seed,S,ymin,ymax):
    ret=[((y-ymin)/(ymax-ymin)-0.5)*2*S for y in seed]
    return ret


N=1000
X=xrange(N)
a=[1,-1,0.01,-0.04]
b=[249.328964117,-1,604.535897664,32,976]

Y=[0,0,0,0]
Y[0]=lineFun(a[0],b[0],X)
Y[1]=lineFun(a[1],b[1],X)
Y[2]=expFun(a[2],b[2],X)
Y[3]=expFun(a[3],b[3],X)


plt.figure(1)
plt.subplot(221)
plt.plot(X,Y[0])
plt.subplot(222)
plt.plot(X,Y[1])
plt.subplot(223)
plt.plot(X,Y[2])
plt.subplot(224)
plt.plot(X,Y[3])
#plt.show();
ymin=0
ymax=1
r=np.random.rand(1,N)
X=[x*10 for x in r]
plt.figure(2)
plt.plot(range(1000),r[0])

S1=9
r1=[x*9 for x in r[0]]
plt.figure(3)
plt.plot(range(1000),r1)
ymax=S1*ymax

S2=5
plt.figure(4)
r2=transform(N,r1,5,0,9)
plt.plot(range(1000),r2)
plt.show()

