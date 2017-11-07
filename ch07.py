import numpy as np
import matplotlib.pyplot as plt
import math

def getRandomND(N):
    return (np.random.normal(size=(N,1)))
def getMyRandom(N):
    rn=getRandomND(N)
    return np.sin(rn)
# Get harmonic signal samples 
# A : amplitude
# f : signal frequency
# dt : 1/fsp duration between samples , fsp : sampling frequency
# N : number of samples
def harm(A,f,dt,N):
    pi=3.1416
    y=A*np.sin(np.array(range(N))*2*pi*f*dt)
    desc="%dsin(2PI%.0ft)"%(A,f)
    y=y.reshape((N,1))
    return y,desc
# Shift signal with const value
# X : original signal [vector of numbers]
# const: shifting const value
def shift(X,const):
    return X+const

def spike(sig,num):
    X=sig+0
    n=X.shape[0]
    meanX=X.mean()
    value=np.max(abs(X))
    rnd=np.random.randint(n, size=num)
    for ind in rnd:
        if ind%2==0:
            X[ind]+=10
        else:
            X[ind]-=12
    return X
####forier Transform
# get real part of forier transform
def getRen(xk,n,N):
    PI=3.1416
    return sum([xk[k]*math.cos((2.0*PI*n*k)/N) for k in xrange(N)])/N
def getRe(xk,N):
    return [getRen(xk,n,N) for n in xrange(N)]
# get Imaginary part of forier transform
def getImn(xk,n,N):
    PI=3.1416
    return sum([xk[k]*math.sin((2.0*PI*n*k)/N) for k in xrange(N)])/N
def getIm(xk,N):
    return [getImn(xk,n,N) for n in xrange(N)]
# calculate module of the forier serie 
def getCn(Re,Im,N):
    return [math.sqrt(Re[i]*Re[i]+Im[i]*Im[i]) for i in xrange(N)]
# scale the domain
def getScaledDomain(dt,N):
    return [k*1.0/(dt*N) for k in xrange(N)]
def cutSignal(xk,N1,N):
    xknew=xk[:N1]+[0 for k in xrange(N-N1)]
    return xknew



# First create three harmonic signals
N=1000
dt=0.001
# 100sin(2PI53t)
A=100
c=A*10
f=53
xk,desc1=harm(A,f,dt,N)
xk=shift(xk,c)

re=getRe(xk,N)
im=getIm(xk,N)
Cn=getCn(re,im,N)
Cn=Cn[:N/2]

f=getScaledDomain(dt,N)

plt.figure(1)
plt.plot(xk)
plt.xlabel("time(n)")
plt.ylabel("f(n)")
#plt.show()

plt.figure("Fourier transform")
plt.plot(Cn);
plt.xlabel("frequency(k)")
plt.ylabel("Power spectrum")
plt.show()

# Antishift
def antiShift(X):
    return X-X.mean()
xk2=antiShift(xk)

re2=getRe(xk2,N)
im2=getIm(xk2,N)
Cn2=getCn(re2,im2,N)
Cn2=Cn2[:N/2]

f=getScaledDomain(dt,N)

plt.figure(5)
plt.subplot(221)
plt.plot(xk)
plt.xlabel("time(n)")
plt.ylabel("f(n)")
plt.subplot(222)
plt.plot(Cn);
plt.xlabel("frequency(k)")
plt.ylabel("Power spectrum")
plt.subplot(223)
plt.plot(xk2)
plt.xlabel("time(n)")
plt.ylabel("f(n)")
#plt.show()
plt.subplot(224)
plt.plot(Cn2);
plt.xlabel("frequency(k)")
plt.ylabel("Power spectrum")
plt.show()

a=0.5
b=100

dt=0.001
xk=np.arange(N)
yk=a*xk+b

plt.figure(2)
plt.plot(yk)

re=getRe(yk,N)
im=getIm(yk,N)
Cn=getCn(re,im,N)
Cn=Cn[:N/2]
plt.figure("Fourier transform2")
plt.plot(Cn);
plt.xlabel("frequency(k)")
plt.ylabel("Power spectrum")
plt.show()

xk=getRandomND(N)
plt.figure(3)
plt.plot(xk)

re=getRe(xk,N)
im=getIm(xk,N)
Cn=getCn(re,im,N)
Cn=Cn[:N/2]
plt.figure("Fourier transform3")
plt.plot(Cn);
plt.xlabel("frequency(k)")
plt.ylabel("Power spectrum")
plt.show()

def plotSpikes(X):
    frame1 = plt.gca()
#    frame1.axes.yaxis.set_ticklabels([])
    plt.stem(X)


xk=np.zeros((N,1))
xk=spike(xk,5)
print xk.shape
plt.figure(4)
plotSpikes(xk)

re=getRe(xk,N)
im=getIm(xk,N)
Cn=getCn(re,im,N)
Cn=Cn[:N/2]
plt.figure("Fourier transform3")
plt.plot(Cn);
plt.xlabel("frequency(k)")
plt.ylabel("Power spectrum")
plt.show()

