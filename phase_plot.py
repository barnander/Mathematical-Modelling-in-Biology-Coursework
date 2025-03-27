#%% import packages
import numpy as np
import matplotlib.pyplot as plt
#%% define ODE
def f(state, t, par):
    gx, gy, a, s, tau = par
    x = state[0]
    y = state[1]
    dxdt = (1 + x**2 + a*s*x**4)/( (1+x**2+s*x**4)*(1+y**4) ) - gx * x
    dydt = 1/tau * ( (1 + x**2 + a*s*x**4)/( (1+x**2+s*x**4)*(1+y**4) ) - gy * y )
    return [dxdt, dydt]
# %% get mesh grid
xlo = 
