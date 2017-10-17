# modeling 
# 1. create three harmonic signal (sampling) fi(n)=Asin(2PI*f*k*dt)
# 2. Add these signals to get single poly harmonic signal  f(n)=f1(n)+f2(n)+f3(n)
# 3. Create random spikes in the signal
# 4. shift the signal with const value
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

# First create three harmonic signals
N=1000
dt=0.001
# 100sin(2PI53t)
A=100
f=53
y1,desc1=harm(A,f,dt,N)
# 25sin(2PI5t)
A2=25
f2=5
y2,desc2=harm(A2,f2,dt,N)
# 30sin(2PI180t)
A3=30
f3=180
y3,desc3=harm(A3,f3,dt,N)
# Plot the three harmonic signals
plt.figure("Generate three harmonic signals",figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(3,1,1)
plt.plot(range(N), y1, 'r-')
plt.title("f1(t)=%s;dt=%.3f"%(desc1,dt))
plt.xlabel('n')
plt.ylabel('f(n)')
plt.subplot(3,1,2)
plt.plot(range(N), y2, 'r-')
plt.title("f2(t)=%s;dt=%.3f"%(desc2,dt))
plt.xlabel('n')
plt.ylabel('f(n)')
plt.subplot(3,1,3)
plt.plot(range(N), y3, 'r-')
plt.title("f3(t)=%s;dt=%.3f"%(desc3,dt))
plt.xlabel('n')
plt.ylabel('f(n)')
plt.tight_layout()
plt.savefig('figures/c04-01-Generate three harmonic signals.png')
# Second: Create signal poly harmonic signal
y=[y1[i]+y2[i]+y3[i] for i in range(N)]
plt.figure("Add three harmonic signals")
plt.plot(range(N),y,'r-')
plt.title("f(t)=%s+%s+%s;dt=%.3f"%(desc1,desc2,desc3,dt))
plt.xlabel('n')
plt.ylabel('f(n)')
plt.tight_layout()
plt.savefig('figures/c04-02-Add three harmonic signals.png')
plt.show()
# Third: Create spikes in the signal
ysp=spike(y,5)
plt.figure("Add random spikes to the signal")
plt.plot(range(N),ysp,'r-')
plt.title("Signal after adding random spikes")
plt.xlabel('n')
plt.ylabel('f(n)')
plt.savefig('figures/c04-03-Add random spikes to the signal')
plt.show()

# Forth: shift the signal with const value
const=100
ysh=shift(y,const)
plt.figure("Shifting signal by constant value")
plt.plot(range(N),ysh,'r-')
plt.title("Shifting signal by %d"%const)
plt.xlabel('n')
plt.ylabel('f(n)')
plt.savefig('figures/c04-04-Shifting signal by constant value')
plt.show()
