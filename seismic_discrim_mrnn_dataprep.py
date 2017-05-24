import numpy as np
import os
import obspy as op
import scipy as sp
import matplotlib.pyplot as plt

#copyright Cooper W. Harris, University of Southern California, 2 May 2017

#first get the explosion data ready
os.chdir('../Data/Explosion')

b = op.read('*SAC')
bcount = len(b)

target = 2.0

bnpts = len(b[0])

T = np.zeros((bcount,bnpts))
R = np.zeros((bcount,bnpts))
d = []

for i in range(bcount):
    T[i] = b[i]
    T[i] = T[i]/np.linalg.norm(T[i])
    snr = float(np.std(T[i,110:300])/np.std(T[i,0:110]))
    unity = T[i]/max(abs(T[i]))
    mean = np.mean(abs(unity))
    if snr >= target:
        R[i]=T[i]
    x = np.linalg.norm(R[i])
    if x == 0:
        d.append((i))

B = sp.delete(R,d,0)


#now get eq data ready
os.chdir('../Earthquake')

e = op.read('*SAC')
e.normalize()

ecount = len(e)
enpts = len(e[0])

G = np.zeros((ecount,enpts))
S = np.zeros((ecount,enpts))
p = []

for i in range(ecount):
    G[i] = e[i]
    G[i] = G[i]/np.linalg.norm(G[i])
    snr = float(np.std(G[i,110:300])/np.std(G[i,0:110]))
    if snr >= target:
        S[i]=G[i]
    x = np.linalg.norm(S[i])
    if x == 0:
         p.append((i))

E = sp.delete(S,d,0)

#stack the trace arrays together to get a feature array

X = np.vstack((B,E))

#now make two label arrays and stack them IN SAME ORDER
num_b = B.shape[0]
num_e = E.shape[0]
by = np.ones((num_b,1)) 
ey = np.zeros((num_e,1))

#in L: 1=explosion, 0=earthquake
L = np.vstack((by,ey))
l = len(L)

#in Q: 0=explosion, 1=earthquake
Q = np.ones([l,1]) - L

#stack dummy matrices L, Q horizontally to make the one-hot encoded label matrix
Y = np.hstack((L,Q))

#print out how many traces passed the criterion
print(str(num_b), "explosions")
print(str(num_e), "earthquakes")

#now save the files

os.chdir('../')

np.savetxt('X_feature_' + str(target) + 'snr.csv',X,delimiter=',')
np.savetxt('Y_label_' + str(target) + 'snr.csv',Y,delimiter=',')


