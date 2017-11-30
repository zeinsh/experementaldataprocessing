
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


def getRandomND(N): #Normal distribution
    return (np.random.normal(size=(N,1)))
def getMyRandom(N):
    rn=getRandomND(N)
    return np.sin(rn)


# In[9]:


#moved to statsmd
def getStatistics(X):
    N=X.shape[0]
    xmean=X.mean()
    ck=np.sum(X**2)/float(N)
    ck0=np.sqrt(ck)
    D=np.var(X)
    co=np.std(X)
    M3=np.sum((X-xmean)**3)/float(N)
    gama1=(M3)/np.power(co,3)
    M4=np.sum((X-xmean)**4)/float(N)
    gama2=(M4)/np.power(co,4)-3
    ret={'mean':xmean,
         'ck':ck,
         'ck0':ck0,
         'D':D,
         'co':co,
         'M3':M3,
         'gama1':gama1,
         'M4':M4,
         'gama2':gama2
        }
    return ret
def RxxStep(X,xmean,l):
    n=X.shape[0]
    cntX=X-xmean
    part1=cntX[:n-l]
    part2=cntX[l:n ]
    ret=np.multiply(part1,part2)
    return np.sum(ret)/float(n-l)
def Rxx(X,L):
    xmean=np.mean(X)
    ret=np.array([RxxStep(X,xmean,i) for i in xrange(L)])
    return ret/np.max(abs(ret))
def RxyStep(X,Y,l):
    n=X.shape[0]
    xmean=X.mean()
    ymean=Y.mean()
    cntX=X-xmean
    cntY=Y-ymean
    part1=cntX[:n-l]
    part2=cntY[l:n ]
    ret=np.multiply(part1,part2)
    return np.sum(ret)/float(n-l)
def Rxy(X,Y,L):
    xmean=np.mean(X)
    ymean=np.mean(Y)
    ret=np.array([RxyStep(X,Y,i) for i in xrange(L)])
    return ret/np.max(abs(ret))


# In[4]:


####forier Transform
# get real part of forier transform
def getRen(xk,n,N):
    return sum([xk[k]*math.cos((2.0*np.pi*n*k)/N) for k in xrange(N)])/N
def getRe(xk,N):
    return np.array([[getRen(xk,n,N)] for n in xrange(N)])
# get Imaginary part of forier transform
def getImn(xk,n,N):
    PI=3.1416
    return sum([xk[k]*math.sin((2.0*np.pi*n*k)/N) for k in xrange(N)])/N
def getIm(xk,N):
    return np.array([[getImn(xk,n,N)] for n in xrange(N)])
# calculate module of the forier serie 
def getCn(Re,Im,N):
    return [math.sqrt(Re[i]*Re[i]+Im[i]*Im[i]) for i in xrange(N)]
def fourierTransform(xk,N):
    im=getIm(xk,N)
    re=getRe(xk,N)
    return getCn(re,im,N)


# In[5]:


# scale the domain
def getScaledDomain(dt,N):
    return [k*1.0/(dt*N) for k in xrange(N)]
def cutSignal(xk,N1,N):
    xknew=np.copy(xk)
    xknew[N1:,:]=0
    return xknew


# In[6]:


# Get Signal with random spikes
# X : original signal [vector of numbers]
# num : number of random spikes
def spike(sig,num):
    X=sig+0
    n=X.shape[0]
    meanX=X.mean()
    value=np.max(abs(X))+10
    rnd=np.random.randint(n, size=num)
    dire=np.random.rand(num)
    for i in xrange(num):
        ind=rnd[i]
        if X[ind]>meanX or (X[ind]==meanX and dire[i]>0.5):
            X[ind]+=1.2*value
        else:
            X[ind]-=1.2*value
    return X
# Shift signal with const value
# X : original signal [vector of numbers]
# const: shifting const value
def shift(X,const):
    return X+const


# In[7]:


# Get harmonic signal samples 
# A : amplitude
# f : signal frequency
# dt : 1/fsp duration between samples , fsp : sampling frequency
# N : number of samples
def harm(A,f,dt,N):
    pi=3.1416
    y=A*np.sin(np.arange(N)*2*pi*f*dt)
    desc="%dsin(2PI%.0ft)"%(A,f)
    y=y.reshape((N,1))
    return y,desc


# In[8]:


def getYk_conv(X,h,k):
    """
    get kth value of the convolution
    parameters:
       X : numpy array of shape (N,1)
       h : numpy array of shape (M,1)
       k : scalar integer
    returns:
       kth value of the convolution between X and h
    """
    N=X.shape[0]
    M=h.shape[0]
    num=min(k,M)
    ser=[X[k-l-1]*h[l] for l in xrange(num) if k-l-1<N]
    return sum(ser)
def convolution(X,h):
    """
    get kth value of the convolution
    parameters:
       X : numpy array of shape (N,1)
       h : numpy array of shape (M,1)
    returns:
       the convolution between X and h
    """
    N=X.shape[0]
    M=h.shape[0]
    return np.array([getYk_conv(X,h,k) for k in xrange(N+M)])
# Low pass filter
def lpf(fout,m,dt,lpw):
    #return lpw
    d=[0.35577019,0.24362830,0.07211497,0.00630165]
    arg=2*fout*dt
    lpw[0]=arg
    arg*=np.pi

    for i in range(1,m+1):
        lpw[i]=np.sin(arg*i)/(np.pi*i)
    lpw[m]/=2
    sumy=lpw[0]
    for i in range(1,m+1):
        sum=d[0]
        arg=(np.pi*i)/m

        for k in range(1,4):
            sum+=2*d[k]*np.cos(arg*k)

        lpw[i]*=sum
        sumy+=2*lpw[i]
    #2m+1
    for i in range(m+1):
        lpw[i]/=sumy
    lpw2=np.fliplr([lpw])[0]
    full_lpw=np.concatenate((lpw2[:-1], lpw), axis=0)
    return full_lpw

def readFloatFromDat(filename):
    import numpy as np
    myarray = np.fromfile(filename,dtype=float)
    return myarray

def readFromDat(filename,dtype,count=-1):
    import numpy as np
    myarray = np.fromfile(filename,dtype=dtype,count=count)
    return myarray

# Filters
def lpf(fout,m,dt):
    lpw=np.zeros(m+1)
    d=[0.35577019,0.24362830,0.07211497,0.00630165]
    arg=2*fout*dt
    lpw[0]=arg
    arg*=np.pi

    for i in range(1,m+1):
        lpw[i]=np.sin(arg*i)/(np.pi*i)
    lpw[m]/=2
    sumy=lpw[0]
    for i in range(1,m+1):
        sum=d[0]
        arg=(np.pi*i)/m

        for k in range(1,4):
            sum+=2*d[k]*np.cos(arg*k)

        lpw[i]*=sum
        sumy+=2*lpw[i]
    #2m+1
    for i in range(m+1):
        lpw[i]/=sumy
    lpw2=np.fliplr([lpw])[0]
    full_lpw=np.concatenate((lpw2[:-1], lpw), axis=0)
    return full_lpw

def hpf(fc,m,dt):
    hpw=np.zeros(2*m+1)
    lpw=lpf(fc,m,dt)
    for k in range(2*m+1):
        if k==m:
            hpw[k]=1-lpw[k]
        else:
            hpw[k]=-lpw[k]
    return hpw
def bpf(fc1,fc2,m,dt):
    lpw1=lpf(fc1,m,dt)
    lpw2=lpf(fc2,m,dt)
    return lpw2-lpw1
def bsf(fc1,fc2,m,dt):
    bsw=np.zeros(2*m+1)
    lpw1=lpf(fc1,m,dt)
    lpw2=lpf(fc2,m,dt)
    for k in range(2*m+1):
        if k==m:
            bsw[k]=1+lpw1[k]-lpw2[k]
        else:
            bsw[k]=lpw1[k]-lpw2[k]
    return bsw
