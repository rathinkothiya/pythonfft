import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time



SAMPLE_RATE = 44100  #number of samples per second
N=32768 #window size
data = pd.read_csv('rathin.dat', sep='\s+', header=None, skiprows=2)# read file

xvalue=data[1]#read colmn 1
start=time.time(); #starting timer for calculating execution time
yvalue=data[0] #read coumn 0
xvaluelist=[] 
for i in xvalue:
    xvaluelist.append(i)#append amplitude in list
xvaluelist=xvaluelist[:32768] #number of samples above 32768 are removed
X1 = list()

for i in range(0, N):
    X1.append(np.complex(xvalue[i] * 1, 0))#converting to complex value
 
def fft(X):
    N = len(X)
    if N <= 1:
        return
    #dividing value into 2 part
    even = np.array(X[0:N:2])
    odd = np.array(X[1:N:2])
 
    fft(even)
    fft(odd)
    #applying DFT and combining results
    for i in range(0, N//2):
        t = np.exp(np.complex(0, -2 * np.pi * i / N)) * odd[i]
        X[i] = even[i] + t
        X[N//2 + i] = even[i] - t 

fft(X1)#call function
X=X1

end=time.time() #stopping timer
# Plotting 
_, plots = plt.subplots(2)
#plot in time domain from dat file
plots[0].plot(yvalue[0:1000], xvalue[0:1000])
## Plot in frequent domain
powers_all = np.abs(np.divide(X, N//2))
powers = powers_all[0:N//2]
frequencies = np.divide(np.multiply(SAMPLE_RATE, np.arange(0, N/2)), N)
frequencies=frequencies[0:1024]#value reduced to plot a graph
powers=powers[0:1024]#value reduced to plot a graph

f = open("rathinoutput.text", "w")#creating a file for filling data
for i  in range(0,1023):
    f.write("frequency["+str(i)+"]"+str(frequencies[i])+"\n")
    f.write("power["+str(i)+"]"+str(powers[i])+"\n")

f.close()
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plots[1].plot(frequencies, powers)
## Show plots

print("execution time ="+str(end-start))#calculating execution time
plt.show()