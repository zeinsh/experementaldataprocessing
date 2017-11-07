import numpy as np
import matplotlib.pyplot as plt
import math

def getRandomND(N):
    np.random.seed(42)
    return (np.random.normal(size=N))
def getMyRandom(N):
    rn=getRandomND(N)
    return np.sin(rn)
def plotVector(X,fignum):
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
    xmean=X.mean()
    ymean=Y.mean()
    n=len(X)
    v=sum([(X[k]-xmean)*(Y[k+l]-ymean) for k in range(n-l)])
    return v/float(n-l)
def getStatistics(X):
    xmean=np.sum(X)/float(N)
    ck=(1/float(N))*sum([k*k for k in X])
    ck0=math.sqrt(ck)
    D=(1/float(N))*sum([math.pow((xk-xmean),2) for xk in X])
    co=math.sqrt(D)
    M3=(1/float(N))*sum([math.pow((xk-xmean),3) for xk in X])
    gama1=(M3)/math.pow(co,3)
    M4=(1/float(N))*sum([math.pow((xk-xmean),4) for xk in X])
    gama2=(M4)/math.pow(co,4)-3
    ret={'mean':xmean,'ck':ck,'ck0':ck0,'D':D,'co':co,'M3':M3,'gama1':gama1,'M4':M4,'gama2':gama2}
    return ret
def printReport(rep): 
    xmean=rep['mean']
    ck=rep['ck']
    ck0=rep['ck0']
    D=rep['D']
    co=rep['co']
    M3=rep['M3']
    gama1=rep['gama1']
    M4=rep['M4']
    gama2=rep['gama2']
    print('xmean=%f'%xmean)
    print ('ck=%f'%ck)
    print ('ck0=%f'%ck0)
    print 'D=%f'%D
    print 'c0=%f'%co
    print 'M3=%f'%M3
    print 'gama1=%f'%gama1
    print 'M4=%f'%M4
    print 'gama2=%f'%gama2
def getStatisticsSplitter(X,t):
    n=len(X)
    part=n/t
    ret=[]
    for i in range(t):
        ret.append(getStatistics(X[i*part:(i+1)*part]))
    return ret
def plotComp(comdata):
    import matplotlib.pyplot as plt
    plt.figure(1)
    k=1
    n=330
    for col in comdata[0].keys():
        v=[st[col] for st in comdata]
	plt.subplot(n+k)
	plt.plot(v)
	plt.xlabel(col)
	k=k+1
    plt.show()

N=1000
X=getRandomND(N)
stats=getStatistics(X)
printReport(stats)

xmean=np.sum(X)/float(N)
ck=(1/float(N))*sum([k*k for k in X])
ck0=math.sqrt(ck)
D=(1/float(N))*sum([math.pow((xk-xmean),2) for xk in X])
co=math.sqrt(D)
M3=(1/float(N))*sum([math.pow((xk-xmean),3) for xk in X])
gama1=(M3)/math.pow(co,3)
M4=(1/float(N))*sum([math.pow((xk-xmean),4) for xk in X])
gama2=(M4)/math.pow(co,4)-3

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
rxxs=[r/mx for r in rxxs]

plotVector(rxxs,5)
plt.show()

Y=getRandomND(N)
rxys=[Rxy(X,Y,i) for i in xrange(L)]
my=max([abs(x) for x in rxys])
print 'mx Rxy',my
rxys=[r/my for r in rxys]
plotVector(rxys,6)
plt.show()

print '*'*40
ret=getStatisticsSplitter(X,5)
plotComp(ret)
