
# coding: utf-8

# # Fourier Transfourm
#  1. create harmonic signal
#  2. apply fourier transform
#  3. apply frequency domain scaling
#  4. cut part of the signal
#  5. apply fourier transform and notice frequency leakage
# 

# In[16]:


import matplotlib.pyplot as plt
import numpy as np
import math


# In[2]:


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


# In[17]:


# Get Signal with random spikes
# X : original signal [vector of numbers]
# num : number of random spikes
def spike(sig,num):
    X=sig+0
    n=X.shape[0]
    meanX=X.mean()
    value=np.max(abs(X))
    rnd=np.random.randint(n, size=num)
    for ind in rnd:
        if X[ind]>meanX:
            X[ind]+=1.2*value
        else:
            X[ind]-=1.2*value
    return X
# Shift signal with const value
# X : original signal [vector of numbers]
# const: shifting const value
def shift(X,const):
    return X+const


# In[116]:


####forier Transform
# get real part of forier transform
def getRen(xk,n,N):
    k=range(N)
    sk=np.sin(2.0*np.pi*n*np.array(k).reshape((N,1))/N)
    Xk=np.multiply(xk,sk)
    ret=np.sum(Xk)/N
    return ret
def getRe(xk,N):
    return np.array([getRen(xk,n,N) for n in xrange(N)]).reshape((N,1))
# get Imaginary part of forier transform
def getImn(xk,n,N):
    k=range(N)
    sk=np.sin(2.0*np.pi*n*np.array(k).reshape((N,1))/N)
    Xk=np.multiply(xk,sk)
    ret=np.sum(Xk)/N
    return ret
def getIm(xk,N):
    ret=np.array([getImn(xk,n,N) for n in xrange(N)]).reshape((N,1))
    return ret
# calculate module of the forier serie 
def getCn(Re,Im,N):
    return np.sqrt(Re**2+Im**2)

def plotStepVector(X,y,desc,xlabel="x",ylabel="f(x)"):
    plt.figure(desc)
    for i in X:
	plt.plot([i,i],[0,120])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plotVector(X,desc,xlabel="x",ylabel="f(x)"):
    N=X.shape[0]
    xmean=np.mean(X)
    plt.figure(desc)
#    plt.plot([0,N-1],[xmean,xmean])
    plt.plot(X)    
    plt.title(desc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def getYk(x,h,k,N,M):
    num=min(k,M)
    ser=[x[k-l]*h[l] for l in xrange(num) if k-l<N]
    return sum(ser)
def getY(x,h,N,M):
    return np.array([getYk(x,h,k,N,M) for k in xrange(N+M-1)])
def scaleV(v):
    mx=np.max(v)
    mul=120/mx
    return v*mul
N=1000
dt=0.005
T=5 #sec

domain=range(N)
xt=np.zeros(N)
for i in [200,400,600,800]:
    xt[i]=1
#plotStepVector(domain,xt,"xt","t","xt")

alpha=45
f0=14
t=np.array(domain)*dt
ht=np.multiply(np.sin(2*np.pi*f0*t),np.exp(-alpha*t))
#sn=np.array([np.sin(2*np.pi*f0*tx) for tx in domain])
#ex=np.array([np.exp(-alpha*tx) for tx in domain])
#ht=np.multiply(sn,ex)
#plotVector(ht,"ht","t","h(t)")
#plt.show()

M=200
htM=ht[:M]
htM=scaleV(htM)
plotVector(htM,"ht","t","h(t)")
plt.show()

yk=getY(xt,htM,N,M)
plotVector(yk,"y(t)","t","y(t)")
plt.show()
