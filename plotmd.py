
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np


# In[8]:


def saveFigure(desc,TUT_NUM,FIG_COUNT):
    plt.savefig('figures/c%s-%d-%s'%(TUT_NUM,FIG_COUNT,desc))
def plotVector(X,desc,xlabel="x",ylabel="f(x)",showMean=False,subPlot=False):
    if showMean:
        N=X.shape[0]
        xmean=np.mean(X)
        plt.plot([0,N-1],[xmean,xmean])
    plt.plot(X)
    plt.grid()
    plt.title(desc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
def plotHistogram(X,desc):
    import matplotlib.pyplot as plt
    plt.hist(X)
    plt.title(desc)
    plt.xlabel("x")
    plt.ylabel("f(x)")
def plotDistribution(X,desc):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(color_codes=True)
    sns.distplot(X)
    plt.title(desc)
    plt.xlabel("x")
    plt.ylabel("f(x)")

