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

gxs = [0.05, 0.1, 0.12,0.17, 0.2,0.3]
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
    plt.plot(t, sols[i,:,0], color=color[i], label=str(gx), linewidth=3)
plt.xlabel(r'$t$', fontsize = 30)
plt.ylabel(r'$x$', fontsize = 30)
plt.legend(title=r'$\gamma_x$', title_fontsize=20, fontsize = 14)
plt.tick_params(axis='both', which='major', labelsize=17)
plt.grid(True)
plt.show()
# %%
color = plt.cm.plasma(np.linspace(0, 1, len(gxs)))

for i, gx in enumerate(gxs):
    plt.plot(t, sols[i,:,1], color=color[i], label=str(gx), linewidth=3)
plt.xlabel(r'$t$', fontsize = 30)
plt.ylabel(r'$y$', fontsize = 30)
#plt.legend(title=r'$\gamma_x$', title_fontsize='large')
plt.tick_params(axis='both', which='major', labelsize=17) 
plt.grid(True)
plt.show()

# %%
color = plt.cm.plasma(np.linspace(0, 1, len(gxs)))

for i, gx in enumerate(gxs):
    plt.plot(sols[i,:,0], sols[i,:,1], color=color[i], linewidth=3)

plt.xlabel(r'$x$', fontsize = 30)
plt.ylabel(r'$y$', fontsize = 30)
#explicitly change legend title font size

plt.tick_params(axis='both', which='major', labelsize=17) 
plt.grid(True)

#add stable eq
for i, gx in enumerate(gxs):
    if i == 1:
        legendEq = 'Stable equilibria'
        plt.plot(x0, y0, 'kx', markersize=8, mew = 2, label="Initial conditions")
    else:
        legendEq = None
    if i !=3 and i != 4:
        plt.plot(sols[i,-1,0], sols[i,-1,1], 'ko', markersize=7, mew = 2, label=legendEq)


plt.legend(fontsize = 17)
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
plt.plot(sol1[:,0], sol1[:,1], 'c', linewidth = 3, label='Orbits')
plt.plot(sol1[0,0], sol1[0,1], 'kx', markersize=10, mew = 3, label='Initial conditions')
plt.plot(sol6[:,0], sol6[:,1], 'c', linewidth = 0.75)
plt.plot(sol6[0,0], sol6[0,1], 'kx', markersize=10, mew = 3)
plt.plot(sol4[:,0], sol4[:,1], 'c', linewidth = 2)
plt.plot(sol4[0,0], sol4[0,1], 'kx', markersize=10, mew = 3)


#and stable eq
plt.plot(sol2[-1,0], sol2[-1,1], 'ko', markersize=8, mew = 3, label='Stable equilibrium')
#plot limitcycles
plt.plot(sol5[:,0], sol5[:,1], 'k', linewidth=4, label='Stable Limit cycle')
plt.plot(sol3[:250,0], sol3[:250,1], 'r', linewidth=4, label='Unstable Limit cycle')


plt.xlabel(r'$x$', fontsize = 30)
plt.ylabel(r'$y$', fontsize = 30)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.grid(True)
plt.legend(fontsize = 14)
plt.show()

# %% phase plane for 
gx = 0.17
gy = 0.04
pars = [gx, gy, a, s, tau]

xmin = -0.5
xmax = 3
ymin = 1.75
ymax = 3.5
x_vals = np.linspace(xmin,xmax, 100)
y_vals = np.linspace(ymin,ymax, 100)
X, Y = np.meshgrid(x_vals, y_vals)
U, V = f([X.ravel(), Y.ravel()], 0, pars)

# Plot the streamplot (flow plot)
plt.figure(figsize=(7,7))
plt.streamplot(X, Y, U.reshape((100,100)), V.reshape((100,100)), color='blue', linewidth=1)


# %%
def phasePlane(pars,xBounds,yBounds, x0s,t, ax, LC = False, nx = 100, ny = 100, stream = True, streamDensity = 1, nSkip = 1, quivScale = 1, legend = False):
    xLo, xHi = xBounds
    yLo, yHi = yBounds
    x_vals = np.linspace(xLo, xHi, nx)
    y_vals = np.linspace(yLo, yHi, ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    uDot,vDot = f([X.ravel(),Y.ravel()], 0, pars)
    eqId = np.argmin(np.abs(uDot)**2.2 + np.abs(vDot)**2) 


    eqX = X.ravel()[eqId]
    eqY = Y.ravel()[eqId]
    ax.set_xlim(xLo, xHi)
    ax.set_ylim(yLo, yHi)
    # plt.xlim(xLo, xHi)
    # plt.ylim(yLo, yHi)
    U = uDot.reshape((ny,nx))
    V = vDot.reshape((ny,nx))
    if stream:
        ax.streamplot(X, Y, U, V, color='k', linewidth=1, density = streamDensity)
    else:
        ax.quiver(X[::nSkip,::nSkip], Y[::nSkip,::nSkip], U[::nSkip,::nSkip], V[::nSkip,::nSkip], color='blue', scale = quivScale)
    ax.contour(X, Y, U, levels=[0], colors='r', linewidths=8, label = r"$\dot{x} = 0$")
    ax.contour(X, Y, V, levels=[0], colors='g', linewidths=8, label = r"$\dot{y} = 0$")
    # for x0 in x0s:
    #     sol = odeint(f, x0, t, args=(pars,))
    #     ax.plot(sol[:,0], sol[:,1], 'k--', linewidth = 3, label='Orbit')
    x0 = x0s[0]
    sol = odeint(f, x0, t, args=(pars,))
    if LC:
        ax.plot(sol[-500:,0], sol[-500:,1], 'k', linewidth = 3, label='Limit Cycle')
        ax.plot(eqX, eqY, 'o', markeredgewidth = 4,  markeredgecolor = 'k', markerfacecolor= 'none', markersize=12, label='Unstable equilibrium')

    else:
        ax.plot(sol[-1,0], sol[-1,1],'ko', markersize=12, mew = 3, label='Unstable equilibrium')

    labels = [ r"$\frac{dx}{dt} = 0$", r"$\frac{dy}{dt} = 0$", "Stable LC", "Stable Eqm", "Unstable Eqm" ] 
    legend_handles = [plt.Line2D([0], [0], color='r', linestyle='-', linewidth = 5),
                      plt.Line2D([0], [0], color='g', linestyle='-', linewidth = 5),
                      plt.Line2D([0], [0], color='k', linestyle='-', linewidth = 2),
                      plt.Line2D([0], [0], marker='o', color='k', markerfacecolor='k', markeredgecolor='black', linestyle='None', markersize = 10),
                      plt.Line2D([0], [0], marker='o', color='k', markerfacecolor='none', markeredgecolor='black', linestyle='None', markersize = 10)]
    if legend:
        ax.legend(legend_handles, labels, fontsize = 20, loc = 'lower right')
    #find equilibrium
    ax.set_xlabel("$x$", fontsize = 30)
    ax.set_ylabel("$y$", fontsize = 30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    return 


# %%
T = 2000
t = np.linspace(0, T, 1000)
gx = 0.17
gy = 0.05
pars = [gx, gy, a, s, tau]
LC = True

fig, ax = plt.subplots(figsize=(7,7))
xmin = -0.5
xmax = 3.75
ymin = 1.75
ymax = 3.5
x0s = [[0,1.76],[3,3.4],[3.5,2],[0.55,2.18],[-0.4,3.2],[1.5,1.8]]
phasePlane(pars, [xmin,xmax], [ymin,ymax],x0s, t, ax, nx = 50, ny = 50, streamDensity = 2.5, nSkip=3, quivScale=1.5, LC = LC)


plt.show()

# %%
gxs = [0.05, 0.1, 0.12,0.17, 0.2, 0.25]
gy = 0.04
a = 11
s = 2
tau = 5
xBounds = [-0.5,3.6]
yBounds = [1.75,3.3]
x0 = [0,2]


for i, gx in enumerate(gxs):
    pars = [gx, gy, a, s, tau]
    plt.subplot(1,len(gxs),i+1)
    phasePlane(pars, xBounds, yBounds,50,50, streamDensity=1, stream=True)
    # run simulation x0
    sol = odeint(f, x0, np.linspace(0,2000,50000), args=(pars,))
    plt.plot(sol[:,0], sol[:,1], 'k', linewidth = 3, label='Orbit')
plt.show()
# %%
