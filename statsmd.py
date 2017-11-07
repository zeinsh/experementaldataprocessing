
# coding: utf-8

# In[6]:


import numpy as np


# In[4]:


def getStatistics(X):
    import numpy as np
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
def printStatsReport(rep):
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
def splittedDomainStats(X,t):
    n=len(X)
    part=n/t
    ret=[]
    for i in range(t):
        ret.append(getStatistics(X[i*part:(i+1)*part]))
    return ret


# In[5]:


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

