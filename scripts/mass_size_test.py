# ---user modules

import GCevo.config as cfg
import GCevo.profiles as pr
from GCevo.orbit import orbit
import GCevo.evolve as ev


# ---python modules
import numpy as np
import time

# ---for plot
import matplotlib as mpl  # must import before pyplot

mpl.use('Qt5Agg')
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 15
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import seaborn as sns

def mass_size(M):
    R4=2.548 * 1e-3
    beta=0.242
    return R4 * (M / 1e4) ** beta


def ftidal(x, ft1=0.77, eta=0.19):
    """
    Calibrated to the Penarrubia+10 tidal track, as done in
    test_evolve_StarCluster_20220717CalibrateftUsingTidalTrack.py
    """
    return ft1 * x ** eta
# ---for evolution and bookkeeping
Nstep = 200  # number of timesteps
tmax = 10.  # [Gyr]
timesteps = np.linspace(1e-6, tmax, Nstep)  # [Gyr]
def fcolor(v, vmin=0., vmax=tmax, choice='t'):
    r"""
    Returns

        - a color,
        - the color map from which the color is drawn,
        - the normalization of the color map,

    upon input of a property.

    Syntax:

        fcolor(v,vmin=vmin,vmax=vmax,choice='...')

    where

        v: the value of the halo property of choice
        vmin: the value below which there is no color difference
        vmax: the value above which there is no color difference
        choice: ...
    """
    if choice == 't':
        cm = plt.cm.coolwarm
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        return scalarMap.to_rgba(v), cm, norm

# ---numerical resolution
cfg.Mres = 100.  # [M_sun]
cfg.Rres = 1e-2  # [kpc]

# ---dynamical friction
# cfg.lnL_type = 0  # <<< Coulomb log choice, 0 = fiducial: lnL = ln(M/m)
# cfg.lnL_type = 1  # <<< Coulomb log choice, 1 = Petts+15/16
# cfg.lnL_pref = 1. # <<< empirical pre-factor for the Coulomb log

cfg.lnL_type = 1  # <<< Coulomb log choice, 0 = Bar+22

# ---tidal heating and evaporation efficiencies
fr = 0.08
StrippingEfficiency = 0.55  # default value (as calibrated for subhalos)
# StrippingEfficiency = 1e-6 # test 0 = turn off tidal stripping
xie = 0.0074  # default is 1/137 = 0.0074 for isolated relaxed cluster

# ---host properties
Mh = 10. ** 9  # halo virial mass [M_sun]
ch = 10.
#
Mb = 10 ** 8.3343  # baryon mass [M_sun]
cb = 6.1650  # Burkert-like concentration
#
h = pr.NFW(Mh, ch, Delta=200., z=0.)
Bur = pr.Burkert_like(Mb, cb)

# ---choose the potential to study
potential = [Bur, h]
R0=5.
N=30
xv0 = np.array([np.repeat(R0,N),np.zeros(N),np.zeros(N),np.zeros(N),pr.Vcirc(potential,np.repeat(R0,N))* 0.999,np.zeros(N)]).T
lgMmin = 3.5
lgMmax = 7.5
m0 = np.logspace(lgMmin,lgMmax,N)
lhalf0 = mass_size(m0)
eta = 2.  # half of EFF outer slope
a0 = lhalf0 / (np.sqrt(2. ** (2. / (2 * eta - 3.)) - 1.))  # EFF scale radius

############################### compute #################################

print('>>> initializing ... ')
mass = np.zeros((Nstep + 1, N)) - 99.
HalfMassRadius = np.zeros((Nstep + 1, N)) - 99.
clusters = [0] * N
orbits = [0] * N
print('>>> evolving ... ')
t1 = time.time()
for i in range(Nstep):
    t = timesteps[i]
    if i == 0:  # for the initial step
        dt = t
    else:
        dt = t - timesteps[i - 1]
    for j in range(N):
        if i == 0:  # initialize if at the first step
            m = m0[j]
            a = a0[j]
            clusters[j] = pr.EFF(m, a, eta)
            orbits[j] = orbit(xv0[j])
            mass[i, j] = m
            HalfMassRadius[i, j] = pr.EFF(m, a, eta).rhalf
        else:
            pass

        # ---load orbit instance and satellite profile object
        o = orbits[j]
        s = clusters[j]
        xv = o.xv
        r = np.sqrt(xv[0] ** 2 + xv[2] ** 2)
        m = s.Mh

        # ---evolve orbit
        if r > cfg.Rres:
            o.integrate(dt, potential, satellite=s)
            xv = o.xv
            # note that the coordinates are updated internally in the orbit
            # instance "o", here we assign them to xv only for bookkeeping
        elif m < cfg.Mres or s.rhalf > 0.1:
            xv = o.xv
        elif r < cfg.Rres and m > cfg.Mres and s.rhalf < 0.1:
            # i.e., the satellite has merged to its host, so
            # no need for orbit integration; to avoid potential
            # numerical issues, we assign a dummy coordinate that
            # is almost zero but not exactly zero
            xv = np.array([cfg.Rres, 0., 0., 0., 0., 0.])

        # ---evolve satellite mass and profile
        if r > cfg.Rres and m > cfg.Mres and s.rhalf < 0.1:
            ft = ftidal(m / m0[j])
            s, lt = ev.EFF(s, potential, dt, xv, ft=ft, fr=fr, xie=xie, alpha=StrippingEfficiency, choice='King62')
        else:
            pass
        m = s.Mh
        lhalf = s.rhalf
        mass[i + 1, j] = m
        HalfMassRadius[i + 1, j] = lhalf
        # ---update satellite profile for the next time step
        #   Note that there is no need to update the orbit object since
        #   the coordinates are updated internally
        orbits[j] = orbit(xv)
        clusters[j] = s
t2 = time.time()
print("cost time: " +str(t2-t1)+"s")
print(np.log10(mass[-1]))
print(np.log10(HalfMassRadius[-1]))
skill=np.arange(0,201,5)
skill2=np.arange(0,201,20)
timesteps=np.insert(timesteps,0,0)

mass_result=mass[skill]
lhalf_result=HalfMassRadius[skill]
time_grid=timesteps[skill]



fig,ax=plt.subplots(1,1,figsize=(6,4),dpi=200)
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params('both', direction='in', top='on', right='on', length=10,
                     width=1, which='major')
ax.tick_params('both', direction='in', top='on', right='on', length=5,
                     width=1, which='minor')
ax.set_xlim(5e2,5e7)
ax.set_ylim(1e-3,5e-2)
ax.set_xlabel("$m$ [$M_\odot$]")
ax.set_ylabel(r"$l_{\rm 1/2}$ [kpc]")

line_grid=np.logspace(np.log10(1e2),np.log10(9e7),20)

color=fcolor(time_grid)[0]
for i in range(len(mass_result)):
    ax.scatter(mass_result[i],lhalf_result[i],facecolor=color[i],s=20,alpha=0.5)
ax.plot(line_grid,mass_size(line_grid),color="blue",ls="--",label=r"$l_{\rm 1/2}=2.55(\frac{m}{10^4 M_\odot})^{0.24}$")
ax.legend(loc="best")
axadd = fig.add_axes([0.855, 0.185, 0.025, 0.71])
# cmap = mpl.colors.ListedColormap(c)
norm = mpl.colors.Normalize(vmin=0, vmax=tmax)
cb = mpl.colorbar.ColorbarBase(axadd, cmap=plt.cm.coolwarm, ticks=[0,1,2,3,4,5,6,7,8,9,10], norm=norm,orientation='vertical')
cb.set_label('$t$ [Gyr]',)
plt.tight_layout(rect=[0.00, 0.001, 0.85, 0.95])
plt.savefig("./mass-size_R0=%.2f_Mh=%.2f.pdf"%(R0,np.log10(Mh)))
