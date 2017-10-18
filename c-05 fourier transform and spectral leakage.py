# Fourier Transfourm
# 1. create harmonic signal
# 2. apply fourier transform
# 3. apply frequency domain scaling
# 4. cut part of the signal
# 5. apply fourier transform and notice frequency leakage
import matplotlib.pyplot as plt
import numpy as np
import math

def getRandomND(N):
    return (np.random.normal(size=N))
def getMyRandom(N):
    rn=getRandomND(N)
    r=[math.sin(x) for x in rn]
    return np.array(r)
# Get harmonic signal samples 
# A : amplitude
# f : signal frequency
# dt : 1/fsp duration between samples , fsp : sampling frequency
# N : number of samples
def harm(A,f,dt,N):
    from math import sin
    pi=3.1416
    y=[A*sin(2*pi*f*k*dt) for k in xrange(N)]
    desc="%dsin(2PI%.0ft)"%(A,f)
    return y,desc
# Get Signal with random spikes
# X : original signal [vector of numbers]
# num : number of random spikes
def spike(X,num):
    X1=np.array(X)
    meanX=X1.mean()
    value=max(X1.max(),abs(X1.min()))
    import random
    r=[random.randint(0,len(X)-1) for i in xrange(num)]
    for i in range(num):
	if X1[r[i]]>meanX:
	    X1[r[i]]+=1.2*value
        else:
	    X1[r[i]]-=1.2*value
    return X1
# Shift signal with const value
# X : original signal [vector of numbers]
# const: shifting const value
def shift(X,const):
    return [y+const for y in X]
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

f0=53	  #signal frequency
A0=100    #amplitude
dt=0.001  #sampling time
fgr=1/(2.0*dt)   #sampling frequency

N=2000    # number of samples
N1=1800   # first N1 sample will be kept after cutting the signal , the last N-N1 will be set to 0
T=1       # signal duration

# first : create harmonic signal
xk,desc=harm(A0,f0,dt,N)
# second : apply Fourier transform
re=getRe(xk,N)
im=getIm(xk,N)
Cn=getCn(re,im,N)
# third : domain scaling
f=getScaledDomain(dt,N)
# the first N/2 is enough to represent the spectrum
Cn=Cn[:N/2]
f=f[:N/2]

plt.figure(desc)
plt.title("Harmonic signal f(t)="+desc)
plt.plot(range(N),xk)
plt.xlabel("time(n)")
plt.ylabel("f(n)")
plt.savefig("figures/c05-0%d-%s.png"%(1,"Harmonic signal"))
plt.figure("Fourier transform")
plt.title("Frequency spectrum of the signal : f(t)="+desc)
plt.plot(f,Cn);
plt.xlabel("frequency(k)")
plt.ylabel("Power spectrum")
plt.savefig("figures/c05-0%d-%s.png"%(2,"Frequency spectrum"))
plt.show()

# Fourth : cut the signal 
# keep first N1 , and set the remains to 0
xkcut=cutSignal(xk,N1,N)
# Apply fourier transform
re=getRe(xkcut,N)
im=getIm(xkcut,N)
Cn=getCn(re,im,N)
# domain scaling
f=getScaledDomain(dt,N)
# keep first N/2 frequencies
Cn=Cn[:N/2]

f=f[:N/2]
plt.figure(desc+" , after cutting the last %d samples"%(N-N1))
plt.title("Harmonic signal f(t)="+desc)
plt.plot(range(N),xkcut)
plt.xlabel("time(n)")
plt.ylabel("f(n)")
plt.savefig("figures/c05-0%d-%s.png"%(3,"harmonic signal after curring"))
plt.figure("Spectral Leakage")
plt.title("Frequency spectrum after cutting , Spectral leakage")
plt.plot(f,Cn);
plt.xlabel("frequency(k)")
plt.ylabel("Power spectrum")
plt.savefig("figures/c05-0%d-%s.png"%(4,"spectral leakage"))
plt.show()
