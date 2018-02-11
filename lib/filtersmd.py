import pandas as pd
import numpy as np
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
