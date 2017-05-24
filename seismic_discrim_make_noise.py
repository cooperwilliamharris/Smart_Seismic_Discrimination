import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy
import scipy.signal as sig
import os

#copyright Cooper W. Harris, University of Southern California, 2 May 2017


#make noisy data to add a third class to seismic discrimination mrnn
#all traces have been demeaned and normalized to +/- unity

###data parameters###
#length of traces (number of points)
npts = 996
m = npts
#number of traces
traces = 5
n = traces

#gaussian noise matrix (brownian)
gauss = np.random.normal(0,1,[n,m])
for i in range(n):
#    gauss[i]= gauss[i]-np.mean(gauss[i])
    gauss[i] = gauss[i]/np.linalg.norm(gauss[i])

plt.figure(1)
for i in range(2):
    ax = plt.subplot(6,1,i+1)
    ax.set_xlabel("gauss")
    ax.set_ylim(-1,1)
    plt.plot(range(npts),gauss[i],color='g')

#cauchy noise matrix
cauchy = np.random.standard_cauchy([n,m])
for i in range(n):
#    cauchy[i] = cauchy[i]-np.mean(cauchy[i])
    cauchy[i] = cauchy[i]/np.linalg.norm(cauchy[i])    

for i in range(2,4):
    ax = plt.subplot(6,1,i+1)
    ax.set_xlabel("cauchy")
    ax.set_ylim(-1,1)
    plt.plot(range(npts),cauchy[i-3],color='r')

#poisson spiked brownian noise (levy process) 
#spike = levy.rvs(size=[n,m])
spike = np.random.poisson(lam=3,size=[n,m])
spike = spike + np.random.normal(0,1,[n,m])
for i in range(n):
#    spike[i] = spike[i]-np.mean(spike[i])
    spike[i] = spike[i]/np.linalg.norm(spike[i])
                          
for i in range(4,6):
    ax = plt.subplot(6,1,i+1)
    ax.set_xlabel("poisson")
    ax.set_ylim(-1,1)
    plt.plot(range(npts),spike[i-6],color='b')

plt.tight_layout()
plt.show()



                          
