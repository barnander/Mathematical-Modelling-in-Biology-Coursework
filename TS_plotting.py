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
gxs = [0.05,0.11,0.17, 0.2]
T = 1000
t = np.linspace(0, T, 1000)
x0 = 0
y0 = 2
x = [x0, y0]
sols = np.zeros((len(gxs), len(t), 2))
for i, gx in enumerate(gxs):
    par = [gx, gy, a, s, tau]
    sol = odeint(f, x, t, args=(par,))
    sols[i,:,:] = sol


#%% plot x time series for increasing values of gx

color = plt.cm.plasma(np.linspace(0, 1, len(gxs)))

for i, gx in enumerate(gxs):
    plt.plot(t, sols[i,:,0], color=color[i], label=str(gx), linewidth=2)
plt.xlabel(r'$t$', fontsize = 20)
plt.ylabel(r'$x$', fontsize = 20)
plt.legend(title=r'$\gamma_x$', title_fontsize='large')
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.grid(True)
plt.show()
# %%
color = plt.cm.plasma(np.linspace(0, 1, len(gxs)))

for i, gx in enumerate(gxs):
    plt.plot(t, sols[i,:,1], color=color[i], label=str(gx), linewidth=2)
plt.xlabel(r'$t$', fontsize = 20)
plt.ylabel(r'$y$', fontsize = 20)
plt.legend(title=r'$\gamma_x$', title_fontsize='large')
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.grid(True)
plt.show()

# %%
color = plt.cm.plasma(np.linspace(0, 1, len(gxs)))

for i, gx in enumerate(gxs):
    plt.plot(sols[i,:,0], sols[i,:,1], color=color[i], label=str(gx), linewidth=2)

plt.xlabel(r'$x$', fontsize = 20)
plt.ylabel(r'$y$', fontsize = 20)
plt.legend(title=r'$\gamma_x$', title_fontsize='large')
plt.tick_params(axis='both', which='major', labelsize=14) 
plt.grid(True)

#add stable eq
for i, gx in enumerate(gxs):
    if i !=2:
        plt.plot(sols[i,-1,0], sols[i,-1,1], 'kx', markersize=8, mew = 2)
plt.show()
# %% Let's try to find the hysteresis!!

gx = 0.112304 #halfway between min and max unstable LC
pars = [gx, gy, a, s, tau]
x0 = 0.5
y0 = 2.2
x1 = [x0, y0]

x0 = 0.9507
y0 = 2.678
x2 = [x0, y0]

x0 = 1.17
y0 = 2.678
x3 = [x0, y0]

x0 = 1.2
y0 = 2.678
x4 = [x0, y0]

T = 5000
t = np.linspace(0, T, 10000)
sol1 = odeint(f, x1, np.linspace(0,1000,10000), args=(pars,))
sol2 = odeint(f, x2, t, args=(pars,))
sol3 = odeint(f, x3, t, args=(pars,))
sol4 = odeint(f, x4, t, args=(pars,))

x0 = sol4[-1,0]
y0 = sol4[-1,1]
x5 = [x0, y0]
sol5 = odeint(f, x5, t[:260], args=(pars,))

x0 = 0.9
y0 = 2.6
x6 = [x0, y0]
sol6 = odeint(f, x6, t, args=(pars,))

plt.figure()
plt.plot(sol1[:,0], sol1[:,1], 'c', linewidth = 0.75, label='Orbits')
plt.plot(sol1[0,0], sol1[0,1], 'bx', markersize=8, mew = 2, label='Initial conditions')
plt.plot(sol6[:,0], sol6[:,1], 'c', linewidth = 0.75)
plt.plot(sol6[0,0], sol6[0,1], 'bx', markersize=8, mew = 2)
plt.plot(sol4[:,0], sol4[:,1], 'c', linewidth = 0.75)
plt.plot(sol4[0,0], sol4[0,1], 'bx', markersize=8, mew = 2)


#and stable eq
plt.plot(sol2[-1,0], sol2[-1,1], 'ko', markersize=8, mew = 3, label='Stable equilibrium')
#plot limitcycles
plt.plot(sol5[:,0], sol5[:,1], 'k', linewidth=3, label='Stable Limit cycle')
plt.plot(sol3[:250,0], sol3[:250,1], 'r', linewidth=3, label='Unstable Limit cycle')


plt.xlabel(r'$x$', fontsize = 20)
plt.ylabel(r'$y$', fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.legend()
plt.show()

# %% phase plane for 
gx = 0.05
pars = [gx, gy, a, s, tau]


x_vals = np.linspace(0.4, 1.6, 100)
y_vals = np.linspace(2.2, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
U, V = f([X, Y], 0, pars)

# Plot the streamplot (flow plot)
plt.figure(figsize=(7,7))
plt.streamplot(X, Y, U, V, color='blue', linewidth=1)

# Labels and title
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Flow Plot of a Damped Oscillator")
plt.show()



# %%
