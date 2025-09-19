import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helperredo import *
from nnearestneighbors import *
from astropy.coordinates import SkyCoord
from astropy import units as u
from datetime import datetime

##n nearest neighbors analysis
print_status("nearest neighbors")
print("20 arcsecond version")
print("9:20 am")
n=10
# read the data
# path to the data files
path = './'

# define output directory path
outputdir = './'

# define filename
galpairs_name = path + 'dist_indices_test.pkl'
pairs=path+"pairs.pkl"

# read unique indices for unique galaxy pairs
uidx_name = path + 'unique_pairs.pkl'
print_status(f"Reading the unique indices from {uidx_name}")
upairs = pd.read_pickle(uidx_name)
# read mask of likely blended galaxies if provided
cfact = 1.5
mask_name = path + str(cfact) + 'x_a1pa2.pkl'

# print status
print_status(f"Reading the data from{pairs}")
pairs=pd.read_pickle(pairs)
pointers= np.array([], dtype=int)
#importing data
ra=pairs['ra'].to_numpy(); dec=pairs['dec'].to_numpy(); f1=pairs['f1'].to_numpy(); f2=pairs['f2'].to_numpy()
e1 = pairs['e1'].to_numpy(); e2 = pairs['e2'].to_numpy()
idx1=upairs['idx1']; idx2=upairs['idx2']
del_y = dec[idx1] - dec[idx2]
del_x = -np.cos((dec[idx1]+dec[idx2])/2/206265)*(ra[idx1] - ra[idx2])
theta = np.sqrt(del_x**2 + del_y**2)
maxi=3000 #cut down on the number of objects
print(maxi)
#only including pairs within range
mask=(theta>1)&(theta<20)&(idx1<maxi)&(idx2<maxi)
print(np.max(idx1), np.max(idx2))
pointers=np.append(pointers, idx1[mask])
pointers=np.append(pointers, idx2[mask])
x=ra[pointers]; y=dec[pointers]; f1=f1[pointers]; f2=f2[pointers]; idx1=idx1[mask]; idx2=idx2[mask]
e1=e1[pointers]; e2=e2[pointers]
x=np.cos(y/20625)*x
print(np.max(idx1), np.max(idx2))
dotprod=np.array([])
dotprodtotal, dotprodnew=dotprod, dotprod
dotprodtotale, dotprode=dotprod, dotprod
idx1=[i for i in range(len(idx1))]; idx2=[j for j in range(len(idx2))]

index=np.array([])
friends=np.array([])
maxdist=20; mindist=1
print(idx1[-1])
numneighs=np.array([])
for i in idx1:
    px1, py1=x[i], y[i]
    obj=(px1, py1)
    fi=np.array([f1[i], f2[i]])
    ei=np.array([e1[i], e2[i]])
    neighbors, ids, dists=n_nearest_neighbors(px1, py1, x, y, n=15, exclude_points=obj, max_distance=maxdist, min_distance=mindist)
    #iterating through nearest neighbor pairs
    for j in idx2:
        if i!=j:
            px2, py2=x[j], y[j]
            pair=(px2, py2)
            ex=[obj, pair]
            neighbors, ids, dists=n_nearest_neighbors(px1, py1, x, y, n=10, exclude_points=obj, max_distance=maxdist, min_distance=mindist)
            numneighs=np.append(numneighs, len(ids))
            #seperating when pair has neighbors and doesn't have neighbors
            if len(ids)>0:
                f1nni=f1[ids]; f2nni=f2[ids]
                e1nni=e1[ids]; e2nni=e2[ids]
                avgf1nni=np.mean(f1nni); avgf2nni=np.mean(f2nni)
                avge1nni=np.mean(e1nni); avge2nni=np.mean(e2nni)
                friends=np.append(friends, len(neighbors))
            else:
                avgf1nni=0; avgf2nni=0; avge1nni=0; avge2nni=0
                friends=np.append(friends, 0)


    #appending to list
    dotprodnew=np.append(dotprodnew, (fi[0])*avgf1nni+(fi[1])*avgf2nni)
    dotprode=np.append(dotprode, (ei[0])*avge1nni+(ei[1])*avge2nni)
    dotprodtotal=np.append(dotprodtotal, fi[0]*avgf1nni+fi[1]*avgf2nni)
    dotprodtotale=np.append(dotprodtotale, ei[0]*avge1nni+ei[1]*avge2nni)
    index=np.append(index, i)
        
    if i%100==0:
        #updates
        print("dotprod of new terms", dotprodnew[-1], dotprode[-1])
        print("converging dotprod", np.mean(dotprodtotal), np.mean(dotprodtotale))
        print('on index', i, 'and', i/len(idx1), "done")
        # dot

print("maxdist", maxdist)
print("Average number of neighbors", np.mean(numneighs))
print("quantity for F", np.mean(dotprodtotal))
print("quantity for eps", np.mean(dotprodtotale))
cumavgs=np.array([]); cumindex=np.array([]); cumsig=np.array([])
cumavgse=np.array([]); cumsige=np.array([])
print_status("cume calc")

#calculating the cumelative values
for i in range(0, len(dotprodnew), 100):
    mask=(index<i)
    avgse=dotprode[mask]
    avgs=dotprodnew[mask]
    cumavge=np.mean(avgse)
    cumavg=np.mean(avgs)
    cumavgse=np.append(cumavgse, cumavge)
    cumavgs=np.append(cumavgs, cumavg)
    cumsige=np.append(cumsige, np.std(avgse))
    cumsig=np.append(cumsig, np.std(avgs))
    cumindex=np.append(cumindex, i)


nbins=20
#calculating not cumelative values
x1,n1,mu1,sig1=binvals(index,dotprodnew, nbins=nbins, log=False) #f
x3, n3, mu3, sig3=binvals(index, dotprode, nbins=nbins, log=False) #eps
x2, n2, mu2, sig2=cumindex, cumindex, cumavgs, cumsig
x4, n4, mu4, sig4=cumindex, cumindex, cumavgse, cumsige

#plotting f
fig,(ax1)=plt.subplots(1)
ax1.errorbar(x1,mu1,sig1/np.sqrt(n1),fmt='o',color='blue', label='avg')
ax1.errorbar(x2, mu2, sig2/np.sqrt(n2), fmt='o', color='red', label='cumulative')
ax1.plot(x1,mu1,color='blue',alpha=0.3)
ax1.plot(x2, mu2, color='red', alpha=0.3)
ax1.plot(x1,x1*0,color='black',alpha=0.3)
#ax1.xlabel('Separation (arcseconds)')
ax1.set_ylabel(r'${\cal F}_i \cdot \bar{\cal F}_{NNi}$')
ax1.set_xlabel("index")
ax1.legend()
fname = 'fsqest.png'
figname = outputdir + fname

plt.savefig(figname, dpi=500, bbox_inches="tight")
plt.close()

#plotting shear
fig,(ax1)=plt.subplots(1)
ax1.errorbar(x3,mu3,sig3/np.sqrt(n3),fmt='o',color='blue', label='avg')
ax1.errorbar(x4, mu4, sig4/np.sqrt(n4), fmt='o', color='red', label='cumulative')
ax1.plot(x3,mu3,color='blue',alpha=0.3)
ax1.plot(x4, mu4, color='red', alpha=0.3)
ax1.plot(x3,x3*0,color='black',alpha=0.3)
#ax1.xlabel('Separation (arcseconds)')
ax1.set_ylabel(r'${\epsilon}_i \cdot \overline{\epsilon}_{NNi}$')
ax1.set_xlabel("index")
ax1.legend()
fname = 'epsqest.png'
figname = outputdir + fname

plt.savefig(figname, dpi=500)
plt.close()
epsys=np.sqrt(np.mean(dotprodtotale))
print(epsys)