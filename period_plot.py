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

#%% TODO isolated fixes for gy = 0.06
dataUnstabHigh = dataUnstabHigh[dataUnstabHigh[:,0]<0.18,:]
# %%
xlims = (0.1,0.32)
ylims = (60,130)

plt.figure()
plt.xlim(xlims)
plt.ylim(ylims)
plt.plot(dataStab[:,0], dataStab[:,1], 'ko')
plt.plot(dataUnstab[:,0], dataUnstab[:,1], 'ro')
plt.plot(dataStabHigh[:,0], dataStabHigh[:,1], 'kx')
plt.plot(dataUnstabHigh[:,0], dataUnstabHigh[:,1], 'rx')
plt.ylabel(r'$T$', fontsize=30)
plt.xlabel(r'$\gamma_x$', fontsize=30)
plt.grid(True)
legend_elements = [
    plt.Line2D([0], [0], marker='.', color='black', linestyle='None', markersize=12, label='Stable LCs, $\gamma_y = 0.04$'),
    plt.Line2D([0], [0], marker='.', color='red', linestyle='None', markersize=12, label='Unstable LCs, $\gamma_y = 0.04$'),
    plt.Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=12, label='Stable LCs, $\gamma_y = 0.06$'),
    plt.Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=12, label='Unstable LCs, $\gamma_y = 0.06$'),
]
plt.legend(handles=legend_elements, fontsize = 12, loc = "upper right")
plt.tick_params(axis='both', which='major', labelsize=18) 
plt.show()

# %%
#%% read data
fileName = 'gy_006/period.dat'
datapd = pd.read_csv(fileName, delim_whitespace=True, comment='%', header=None)
data = np.array(datapd)
# %% remove data with 1 or 2 in row 3 
data = data[data[:,3]>2]
dataStabHigh = data[data[:,3]==3]
dataUnstabHigh = data[data[:,3]==4]
# %%
