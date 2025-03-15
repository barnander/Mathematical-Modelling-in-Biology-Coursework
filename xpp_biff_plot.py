#%% Import Packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Load Data
fileName = 'gy_004/y_gx.dat'
data = pd.read_csv(fileName, delim_whitespace=True, comment='%', header=None)
data = np.array(data)
# %% extract equilibria and LCs
idxEq = data[:,4] == 1
dataEq = data[idxEq,:]
idxLC = data[:,4] == 2
dataLC = data[idxLC,:]

#divide into array for each branch
idxEqUnStab = np.where(dataEq[:,3] == 2)[0][1:] #first eq mislabeled
idxEqStab1 = np.arange(1,idxEqUnStab[0]+1)
idxEqStab2 = np.arange(idxEqUnStab[-1],len(dataEq))

dataEqStab1 = dataEq[idxEqStab1,:]
dataEqUnstab = dataEq[np.concatenate([idxEqUnStab,idxEqStab2[1:]]),:]
dataEqStab2 = dataEq[idxEqStab2[1:],:]

dataLCStab = dataLC[dataLC[:,3] == 3,:]
dataLCUnstab = dataLC[dataLC[:,3] == 4,:]


#%%remove dodgy points 
#TODO: case dependent
#This is for gy_004/x_gx.dat
dataLCStab = dataLCStab[dataLCStab[:,2] < 0.7,:]
dataLCUnstab = dataLCUnstab[dataLCUnstab[:,0]<0.18,:]
#%%remove dodgy points 

#TODO: case dependent
#This is for gy_004/y_gx.dat


#%% Plot bifurcation diagram
plt.figure()
plt.plot(dataEqStab1[:,0],dataEqStab1[:,1], 'k', linewidth=2)
plt.plot(dataEqUnstab[:,0],dataEqUnstab[:,1], 'r', linewidth=2)

plt.plot(dataLCStab[:,0],dataLCStab[:,1],'k.')
plt.plot(dataLCUnstab[1:,0],dataLCUnstab[1:,1], 'r.')

plt.plot(dataLCStab[:,0],dataLCStab[:,2], 'k.')
plt.plot(dataLCUnstab[:,0],dataLCUnstab[:,2], 'r.')

plt.plot(dataEqStab2[:,0],dataEqStab2[:,1], 'k', linewidth=2)

plt.xlabel(r'$\gamma_x$', fontsize=20)
plt.ylabel(r'$y$', fontsize=20)
plt.grid(True)
plt.legend(['Stable Equilibria', 'Unstable Equilibria', 'Stable Limit Cycles', 'Unstable Limit Cycles'])
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.show()




# %% plot gy/gx
fileName = 'gy_004/gy_gx.dat'
data = pd.read_csv(fileName, delim_whitespace=True, comment='%', header=None)
#data = np.array(data)

# %%
