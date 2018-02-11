# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np


def scatterData(X,y,classmap):
    import matplotlib.pyplot as plt
    for c,label in classmap.iteritems():
        pltdata=X[(y==c)]
        plt.scatter(pltdata[:,0], pltdata[:,1] ,label=label) 

def plotDecisionBoundary(w,bias,fig_boundary):
    import matplotlib.pyplot as plt
    from helpersmd import getLine
    xline,yline=getLine(w,bias,fig_boundary)
    plt.plot(xline,yline, 'k-',label="Decision Boundary")

def saveFigure(desc,TUT_NUM,FIG_COUNT):
    plt.savefig('figures/c%s-%d-%s'%(TUT_NUM,FIG_COUNT,desc))

def plotVector(X,desc,xlabel="x",ylabel="f(x)"):
    """
    - plot signal X
    - plot mean of the signal
    parameters:
      X: numpy array of shape (n,1)
      desc: describtion of the plot
      xlabel: label of x axis
      ylabel: label of y axis
    """
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


# In[2]:


def plot2D(X,Y,desc,xlabel="x",ylabel="f(x)"):
    plt.plot(X,Y)
    plt.grid()
    plt.title(desc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

