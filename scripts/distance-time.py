import GCevo.config as cfg
import GCevo.profiles as pr
from GCevo.orbit import orbit
import GCevo.init as init
import GCevo.evolve as ev

import numpy as np
import time
from scipy.special import erf

import matplotlib
import matplotlib.pyplot as plt
plt.style.use(["default"])
matplotlib.rc( 'lines', linewidth=3 )
matplotlib.rc( 'font', family='monospace', weight='normal', size=18 )

c_frame = (0,0,0,0.8)
for tick in 'xtick', 'ytick':
    matplotlib.rc( tick+'.major', width=1.5, size=8)
    matplotlib.rc( tick+'.minor', width=1, size=4, visible=True )
    matplotlib.rc( tick, color=c_frame, labelsize=15, direction='in' )
matplotlib.rc( 'xtick', top=True )
matplotlib.rc( 'ytick', right=True )
matplotlib.rc( 'axes', linewidth=1.5, edgecolor=c_frame, labelweight='normal' )
matplotlib.rc( 'grid', color=c_frame )
matplotlib.rc( 'patch', edgecolor=c_frame )
matplotlib.rc("figure.subplot",wspace= 0.2)
matplotlib.rc("figure.subplot",hspace= 0.2)


def ftidal(x, ft1=0.77, eta=0.19):
    """
    Calibrated to the Penarrubia+10 tidal track, as done in
    test_evolve_StarCluster_20220717CalibrateftUsingTidalTrack.py
    """
    return ft1 * x ** eta
    # return 0.5 # <<< test: constant ft

fr = 0.08
StrippingEfficiency = 0.55  # default value (as calibrated for subhalos)
# StrippingEfficiency = 1e-6 # test 0 = turn off tidal stripping
xie = 0.0074  # default is 1/137 = 0.0074 for isolated relaxed cluster


Nstep = 200  # number of timesteps
tmax = 10.  # [Gyr]
timesteps = np.linspace(1e-6, tmax, Nstep)
def get_one_evolution(lnL_type, orbit_ev=True, mass_ev=True, lh0=None):
    # ---numerical resolution
    cfg.Mres = 100.  # [M_sun]
    cfg.Rres = 1e-2  # [kpc]

    # ---dynamical friction
    # ---different types of DF. Details can be seen in profile.py
    cfg.lnL_type = lnL_type

    # ---host properties
    Mh = 10. ** 9.5  # halo virial mass [M_sun]
    ch = 10. #halo concentration
    #
    h = pr.NFW(Mh, ch, Delta=200., z=0.)

    # ---choose the potential to study
    potential = [h]

    # ---star cluster's initial properties
    # ---Mass initialization
    m0 = 10. ** 5.  # [M_sun]
    if lh0 == None:
        lhalf0 = init.Reff_StarCluster(m0)
    else:
        lhalf0 = init.Reff_StarCluster(m0) * 5
    eta = 2.
    a0 = lhalf0 / (np.sqrt(2. ** (2. / (2 * eta - 3.)) - 1.))
    # a0 = 0.002 # [kpc] <<< test, fix the initial cluster scale size
    s0 = pr.EFF(m0, a0, eta=2.)

    # ---initial orbit control
    R0 = 1 # [kpc] # <<< play with
    phi0 = 0.
    z0 = 0.
    VR0 = pr.Vcirc(potential, np.sqrt(R0 ** 2 + z0 ** 2)) *0
    Vphi0 = np.sqrt(pr.Vcirc(potential, np.sqrt(R0 ** 2 + z0 ** 2)) ** 2 - VR0 ** 2)*0.99
    Vz0 = 0.
    #
    xv0 = np.array([R0, phi0, z0, VR0, Vphi0, Vz0])


    radius = np.zeros(Nstep)
    vR = np.zeros(Nstep)
    vz = np.zeros(Nstep)
    vphi = np.zeros(Nstep)
    print('>>> evolving ... ')
    t1 = time.time()
    for i in range(Nstep):
        t = timesteps[i]
        if i == 0:  # for the initial step
            tprevious = 0.
            o = orbit(xv0)
            r = np.sqrt(R0 ** 2 + z0 ** 2)
            m = m0
            s = s0
        else:
            tprevious = timesteps[i - 1]
        dt = t - tprevious
        # ---evolve orbit
        if orbit_ev:
            if r > cfg.Rres:
                o.integrate(t, potential, satellite=s)
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

        r = np.sqrt(xv[0] ** 2 + xv[2] ** 2)

        # ---evolve satellite (star cluster) mass and profile
        if mass_ev:
            if r > cfg.Rres and m > cfg.Mres and s.rhalf < 0.1:
                ft = ftidal(m / m0)
                s, lt = ev.EFF(s, potential, dt, xv, ft=ft, fr=fr, xie=xie, alpha=StrippingEfficiency, choice='King62')
            else:
                pass
        m = s.Mh
        # ---record
        radius[i] = r
    return radius

#Plot distance-time relation for the same GC with different dynamical friction or mass-size evolution
fig,ax=plt.subplots(1,1,figsize=(6,4),dpi=100)

ax.plot(timesteps,get_one_evolution(0),color="blue",ls="-",label="$\ln\Lambda=\ln M/m$",alpha=0.6)
#ax.plot(timesteps,get_one_evolution(0,mass_ev=False),color="blue",ls=":",label="$\ln\Lambda=\ln M/m$ (no mass loss)")
ax.plot(timesteps,get_one_evolution(1),color="black",ls="-",label="Petts+15/16",alpha=0.6)
ax.plot(timesteps,get_one_evolution(1,mass_ev=False),color="red",ls="-",label="Petts+15/16 (no mass loss)",alpha=0.6)
ax.plot(timesteps,get_one_evolution(1,mass_ev=False,lh0=True),color="darkgreen",ls="-",label=r"Petts+15/16 (no mass loss + 5$l_{\rm 1/2,0}$)",alpha=0.6)
ax.plot(timesteps,get_one_evolution(3),color="gray",ls="-",label=r"Bar+22",alpha=0.6)
ax.plot(timesteps,get_one_evolution(4),color="orange",ls="-",label=r"Modak+22",alpha=0.6)
ax.legend(loc="best",borderpad=0,fontsize=12,frameon=False)
ax.set_xlim(0,10)
ax.set_ylim(0,1.2)
ax.set_xlabel("$t$ [Gyr]")
ax.set_ylabel("$R$ [kpc]")
plt.tight_layout(rect=[0.01, 0.01, 1., 1.])
plt.savefig("./distance-time.pdf",bbox_inches='tight')
plt.show()
