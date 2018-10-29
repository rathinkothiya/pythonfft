import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.linalg import dft



SAMPLE_RATE = 44100
N=32768
data = pd.read_csv('rathin.dat', sep='\s+', header=None, skiprows=2)

xvalue=data[1]
xx=xvalue[0:8192]

start=time.time();

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

b=DFT_slow(xx)

end=time.time()

print("execution time in DFT = "+str(end-start))
