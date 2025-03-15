#%% import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#%% define ODE
def f(state, t, par):
    gx, gy, a, s, tau = par
    x = state[0]
    y = state[1]
    dxdt = (1 + x**2 + a*s*x**4)/( (1+x**2+s*x**4)*(1+y**4) ) - gx * x
    dydt = 1/tau * ( (1 + x**2 + a*s*x**4)/( (1+x**2+s*x**4)*(1+y**4) ) - gy * y )
    return [dxdt, dydt]

#%% define parameters
gx = 0.15
gy = 0.04
a = 11
s = 2
tau = 5
par = [gx, gy, a, s, tau]
#%% define initial conditions
x0 = 1
y0 = 1
x = [x0, y0]
#%% define time
T = 500
t = np.linspace(0, T, 1000)
#%% solve ODE
sol = odeint(f, x, t, args=(par,))
#%% plot solution
plt.plot(t, sol[:,0], label='x')
plt.plot(t, sol[:,1], label='y')
plt.xlabel('time')
plt.ylabel('x, y')
plt.legend()



# %% plot TS for increasing values of gx
gxs = np.linspace(0, 0.2, 10)
T = 500

