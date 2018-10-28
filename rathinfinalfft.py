import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
SAMPLE_RATE = 44100
N=32768
data = pd.read_csv('rathin.dat', sep='\s+', header=None, skiprows=2)

xvalue=data[1]
yvalue=data[0]

xvaluelist=[]
for i in xvalue:
    xvaluelist.append(i)
xvaluelist=xvaluelist[:32768]
X1 = list()
for i in range(0, N):
    X1.append(np.complex(xvalue[i] * 1, 0))
 
def fft(X):
    N = len(X)
    if N <= 1:
        return
 
    even = np.array(X[0:N:2])
    odd = np.array(X[1:N:2])
 
    fft(even)
    fft(odd)
 
    for i in range(0, N//2):
        t = np.exp(np.complex(0, -2 * np.pi * i / N)) * odd[i]
        X[i] = even[i] + t
        X[N//2 + i] = even[i] - t

 

fft(X1)
X=X1
# Plotting 
_, plots = plt.subplots(2)
 
## Plot in frequent domain
powers_all = np.abs(np.divide(X, N//2))
powers = powers_all[0:N//2]
frequencies = np.divide(np.multiply(SAMPLE_RATE, np.arange(0, N/2)), N)
frequencies=frequencies[0:1024]
powers=powers[0:1024]

plots[1].plot(frequencies, powers)
## Show plots
plt.show()