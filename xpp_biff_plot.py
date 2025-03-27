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
print(dataLCStab.shape)

#%%remove dodgy points 
#TODO: case dependent
#This is for gy_004/x_gx.dat
dataLCStab = dataLCStab[dataLCStab[:,2] < 0.7,:]
dataLCUnstab = dataLCUnstab[dataLCUnstab[:,0]<0.18,:]
y_lab = r"$x$"
ylims = [0,3.5]
legend = True
#%%remove dodgy points 
#TODO: case dependent
#This is for gy_004/y_gx.dat
dataLCStab = dataLCStab[1:,:]
y_lab = r"$y$"
xlims = [0.05,0.3]
ylims = [1.8,3.2]
legend  = False
#%% #TODO for gy_006/x_gx.dat
dataEqUnstab = np.concatenate([dataEqStab1[-2:,:],dataEqUnstab])
dataEqStab1 = dataEqStab1[:-1,:]
dataLCUnstab = dataLCUnstab[dataLCUnstab[:,0]<0.18,:]
y_lab = r"$x$"
ylims = [0,3.5]
legend  = False
#%% #TODO for gy_006/y_gx.dat
dataEqUnstab = np.concatenate([dataEqStab1[-2:,:],dataEqUnstab])
dataEqStab1 = dataEqStab1[:-1,:]
dataLCUnstab = dataLCUnstab[dataLCUnstab[:,0]<0.18,:]
y_lab = r"$y$"
ylims = [1.8,3.2]
legend  = False
#%%

#%% Plot bifurcation diagram

xlims = (0.05,0.3)
#%% zoom in
xlims = (0.1115,0.116)
ylims = (2,3)
#%%
plt.figure()
LC_weight = 15
plt.xlim(xlims)
plt.ylim(ylims)
plt.plot(dataEqStab1[:,0],dataEqStab1[:,1], 'k', linewidth=4,label='Stable Equilibria')
plt.plot(dataEqUnstab[:,0],dataEqUnstab[:,1], 'r', linewidth=4, label = 'Unstable Equilibria')

plt.plot(dataLCStab[:,0],dataLCStab[:,1],'k.', markersize=LC_weight, label = 'Stable Limit Cycles')
plt.plot(dataLCUnstab[1:,0],dataLCUnstab[1:,1], 'r.', markersize=LC_weight, label = 'Unstable Limit Cycles')

plt.plot(dataLCStab[:,0],dataLCStab[:,2], 'k.', markersize=LC_weight)
plt.plot(dataLCUnstab[:,0],dataLCUnstab[:,2], 'r.', markersize=LC_weight)

plt.plot(dataEqStab2[:,0],dataEqStab2[:,1], 'k', linewidth=4, markersize=10)

plt.plot(dataEqStab1[-1,0],dataEqStab1[-1,1], 'co', markersize=20, mew = 4, markerfacecolor='none', label = "Subcritical Hopf")
plt.plot(dataEqStab2[0,0],dataEqStab2[0,1], 'go', markersize=10, mew = 3, markerfacecolor='none', label = "Supercritical Hopf")

print(r"hopf at $\gamma_x =  $", dataEqStab1[-1,0],dataEqStab1[-1,1])


idxSaddle = np.argmin(dataLCStab[:,0])
plt.plot(dataLCStab[idxSaddle,0],dataLCStab[idxSaddle,1], 'bx', markersize=20, mew = 4, label = "Saddle Node")
plt.plot(dataLCStab[idxSaddle,0],dataLCStab[idxSaddle,2], 'bx', markersize=20, mew = 4)
print(r"Saddle at $\gamma_x =  $", dataLCStab[idxSaddle,0],dataLCStab[idxSaddle,1])
plt.xlabel(r'$\gamma_x$', fontsize=30)
plt.ylabel(y_lab, fontsize=30)
plt.grid(True)
import matplotlib.pyplot as plt

legend_elements = [
    # plt.Line2D([0], [0], color='black', linewidth=3, label='Stable Equilibria'),
    # plt.Line2D([0], [0], color='red', linewidth=3, label='Unstable Equilibria'),
    # plt.Line2D([0], [0], marker='.', color='black', linestyle='None', markersize=12, label='Stable Limit Cycles'),
    #plt.Line2D([0], [0], marker='.', color='red', linestyle='None', markersize=12, label='Unstable Limit Cycles'),
    plt.Line2D([0], [0], marker='o', color='cyan', linestyle='None', markersize=15, fillstyle='none', mew = 5,label='Subcritical Hopf'),
    # #plt.Line2D([0], [0], marker='o', color='green', linestyle='None', markersize=8, fillstyle='none',mew = 3,  label='Supercritical Hopf'),
    plt.Line2D([0], [0], marker='X', color='blue', linestyle='None', markersize=15, label='Saddle Node', mew = 2),
    plt.Line2D([0], [0], color='gray', linestyle='--', linewidth = 4, label=r'$\gamma_x = 0.1123$'),
]
if legend:
    plt.legend(handles=legend_elements, fontsize = 20, loc = "upper right")
#plt.legend(['Stable Equilibria', 'Unstable Equilibria', 'Stable Limit Cycles', 'Unstable Limit Cycles', 'Subcritical Hopf', 'Supercritical Hopf', 'Saddle Node'], fontsize = 15, loc = "upper right")
plt.vlines(0.112304, ymin = 2, ymax = 3, color = 'gray', linestyle = '--', linewidth = 4)
plt.tick_params(axis='both', which='major', labelsize=20) 
plt.show()




# %% plot gy/gx
fileName = 'gy_004/gy_gx.dat'
data = pd.read_csv(fileName, delim_whitespace=True, comment='%', header=None)
#data = np.array(data)

# %%
