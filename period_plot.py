#%% plotting period
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% read data
fileName = 'gy_004/period.dat'
datapd = pd.read_csv(fileName, delim_whitespace=True, comment='%', header=None)
data = np.array(datapd)
# %% remove data with 1 or 2 in row 3 
data = data[data[:,3]>2]
dataStab = data[data[:,3]==3]
dataUnstab = data[data[:,3]==4]

#%% TODO isolated fixes for gy = 0.04
dataStab = dataStab[1:,:]
dataUnstab = dataUnstab[dataUnstab[:,0]<0.16,:]
# %%
plt.figure()
plt.plot(dataStab[:,0], dataStab[:,1], 'ko')
plt.plot(dataUnstab[:,0], dataUnstab[:,1], 'ro')
plt.ylabel(r'$T$', fontsize=20)
plt.xlabel(r'$\gamma_x$', fontsize=20)
plt.grid(True)
plt.legend(['Stable Limit Cycles', 'Unstable Limit Cycles'])
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.show()

# %%
