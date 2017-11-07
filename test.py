import numpy as np
import pandas as pd
from pandas import DataFrame
import math


def getRandomND(N):
    return (np.random.normal(size=N))
def plotVector(X,fignum):
    import matplotlib.pyplot as plt
    N=len(X)
    xmean=np.sum(X)/float(N)
    plt.figure(fignum)
    plt.plot([0,N-1],[xmean,xmean])
    plt.plot(range(N),X)
def plotHistogram(X,fignum):
    import matplotlib.pyplot as plt
    plt.figure(fignum)
    plt.hist(X)
def plotDistribution(X,fignum):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(fignum)
    sns.set(color_codes=True)
    sns.distplot(X)
def plotDeviation(xmean,s,mxv,fignum):
    import matplotlib.pyplot as pl
    xs=[]
    for i in range(-3,4):
        xs.append([xmean-i*s,xmean-i*s])
    y=[0,mxv]
    plt.figure(fignum)
    for x in xs:
        plt.plot(x,y)
def Rxx(X,xmean,l):
    n=len(X)
    v=sum([(X[k]-xmean)*(X[k+l]-xmean) for k in range(n-l)])
    return v/float(n-l)
def Rxy(X,Y,l):
    n=len(X)
    xmean=X.mean()
    ymean=Y.mean()
    v=sum([(X[k]-xmean)*(Y[k+l]-ymean) for k in range(n-l)])
    return v/float(n-l+1)

N=1000
X=getRandomND(N)
Y=getRandomND(N)
Xnp=np.array(X)

h= np.histogram(X)
import matplotlib.pyplot as plt
print len(h[1]),len(h[0])
plt.plot(h[1][:-1],h[0])
plt.show()
xmean=X.mean()
print'xmean',xmean
print 'median',np.median(X)
print 'average',np.average(X)
print 'std',np.std(X)
print 'var',np.var(X)
print 'correlation',np.corrcoef(X[:-10],X[10:])
print 'rxy',Rxy(X[:-10],X[10:],0)
ck=(1/float(N))*sum([k*k for k in X])
print ('ck=%f'%ck)
ck0=math.sqrt(ck)
print ('ck0=%f'%ck0)
D=(1/float(N))*sum([math.pow((xk-xmean),2) for xk in X])
print 'D=%f'%D
co=math.sqrt(D)
print 'c0=%f'%co
M3=(1/float(N))*sum([math.pow((xk-xmean),3) for xk in X])
print 'M3=%f'%M3
gama1=(M3)/math.pow(co,3)
print 'gama1=%f'%gama1
M4=(1/float(N))*sum([math.pow((xk-xmean),4) for xk in X])
print 'M4=%f'%M4
gama2=(M4)/math.pow(co,4)-3
print 'gama2=%f'%gama2
'''
import matplotlib.pyplot as plt
plotVector(X,1)
plotHistogram(X,2)
plotDeviation(xmean,co,350,2)
plotDistribution(X,3)
plt.show()

L=800
rxxs=[Rxx(X,xmean,i) for i in xrange(L)]
mx=max(rxxs)
print 'mx Rxx',mx
rxxs2=[r/mx for r in rxxs]

plotVector(rxxs,5)
plotVector(rxxs2,6)
plt.show()
'''
