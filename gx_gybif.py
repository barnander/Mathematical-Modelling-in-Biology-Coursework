#%%import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
fileName = "gy_gx.dat"
data = pd.read_csv(fileName, delim_whitespace=True, comment='%', header=None)
data = np.array(data)
# %%
data = data[data[:,0] <0.35]
# %% make plot
plt.figure()
plt.plot(data[:,0], data[:,1], 'o')
plt.xlabel(r'$\gamma_x$', fontsize=20)
plt.ylabel(r'$\gamma_y$', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20) 
plt.show()



# %% zoom in on origin
gx_lim  = 0.025
data0 = data[data[:,0] < gx_lim]
x = data0[:,0]
y = m * x + c
plt.figure()
plt.plot(data0[:,0], data0[:,1], 'o')
plt.plot(x, y)
plt.xlabel(r'$\gamma_x$', fontsize=20)
plt.ylabel(r'$\gamma_y$', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()


# %% make seperating line
m = 1/3
x0 = [0.01, 0.001]
c = x0[1] - m*x0[0]
x = data[:,0]
y = m * x + c
plt.figure()
plt.plot(data[:,0], data[:,1], 'o')
plt.plot(x, y)
plt.xlabel(r'$\gamma_x$', fontsize=20)
plt.ylabel(r'$\gamma_y$', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()



# %% divide top and bottom branch
top = data[data[:,1] > m*data[:,0] + c]
bottom = data[data[:,1] < m*data[:,0] + c]

plt.figure()
plt.plot(top[:,0], top[:,1], 'ro')
plt.plot(bottom[:,0], bottom[:,1], 'ko')
#plt.plot(x, y)
plt.xlabel(r'$\gamma_x$', fontsize=30)
plt.ylabel(r'$\gamma_y$', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='r', linestyle='None', markersize=8, mew = 3,label='Subcritical Hopf'),
    plt.Line2D([0], [0], marker='o', color='k', linestyle='None', markersize=8,mew = 3,  label='Supercritical Hopf')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=20)
plt.grid(True)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

plt.figure()

# Plot the points
plt.plot(top[:,0], top[:,1], 'ro')
plt.plot(bottom[:,0], bottom[:,1], 'ko')

# Create continuous curves for filling between
# Sort both arrays by x-values for proper filling
top_sorted = top[top[:,0].argsort()]
bottom_sorted = bottom[bottom[:,0].argsort()]

# Interpolate to get common x-values for filling
x_min = max(top_sorted[:,0].min(), bottom_sorted[:,0].min())
x_max = min(top_sorted[:,0].max(), bottom_sorted[:,0].max())
x_fill = np.linspace(x_min, x_max, 100)

# Interpolate y-values for both curves
top_interp = np.interp(x_fill, top_sorted[:,0], top_sorted[:,1])
bottom_interp = np.interp(x_fill, bottom_sorted[:,0], bottom_sorted[:,1])
plt.fill_between(x_fill, bottom_interp, top_interp, color='gray', alpha=0.3)
mid_x = np.mean(x_fill)
mid_y = np.mean([np.max(top_interp), np.min(bottom_interp)])
plt.text(0.22,0.07, "Limit Cycles", 
         fontsize=24, ha='center', va='center', 
         rotation=30,
         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

plt.text(0.05, 0.5, "Stable \nEquilibria", 
         fontsize=20, ha='left', va='top', transform=plt.gca().transAxes)
plt.text(0.95, 0.05, "Stable\nEquilibria", 
         fontsize=20, ha='right', va='bottom', transform=plt.gca().transAxes)

plt.xlabel(r'$\gamma_x$', fontsize=30)
plt.ylabel(r'$\gamma_y$', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='r', linestyle='None', markersize=8, mew=3, label='Subcritical Hopf'),
    plt.Line2D([0], [0], marker='o', color='k', linestyle='None', markersize=8, mew=3, label='Supercritical Hopf')
]

plt.xlim([0, 0.3])
plt.legend(handles=legend_elements, loc='upper right', fontsize=20)
plt.grid(True)
plt.tight_layout() 
plt.show()
# %%
