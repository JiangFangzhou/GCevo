import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(["default"])

matplotlib.rc('lines', linewidth=3)
matplotlib.rc('font', family='monospace', weight='normal', size=18)

c_frame = (0, 0, 0, 0.8)
for tick in 'xtick', 'ytick':
    matplotlib.rc(tick + '.major', width=1.5, size=8)
    matplotlib.rc(tick + '.minor', width=1, size=4, visible=True)
    matplotlib.rc(tick, color=c_frame, labelsize=15, direction='in')
matplotlib.rc('xtick', top=True)
matplotlib.rc('ytick', right=True)
matplotlib.rc('axes', linewidth=1.5, edgecolor=c_frame, labelweight='normal')
matplotlib.rc('grid', color=c_frame)
matplotlib.rc('patch', edgecolor=c_frame)
matplotlib.rc("figure.subplot", hspace=0.27)
matplotlib.rc("figure.subplot", wspace=0.27)

import numpy as np
import time
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad

import GCevo.config as cfg
import GCevo.profiles as pr
from GCevo.orbit import orbit
import GCevo.evolve as ev
import GCevo.init as init

# Load the data for NGC5846-UDG1
print("begin load data")
data = pd.read_csv(r"./Data.txt", sep=";")
UDG_1 = data.loc[data['host'] == "NGC5846-UDG1-1"]
UDG_2 = data.loc[data['host'] == "NGC5846-UDG1-2"]
UDG_1_Rc = UDG_1["Rg"].to_numpy()
UDG_1_Mc = UDG_1["Mc"].to_numpy()
UDG_2_Rc = UDG_2["Rg"].to_numpy()
UDG_2_Mc = UDG_2["Mc"].to_numpy()
UDG_1_rh = UDG_1["rh"].to_numpy()
UDG_2_rh = UDG_2["rh"].to_numpy()


skill1 = np.where(UDG_1_Rc <= 5)
skill2 = np.where(UDG_2_Rc <= 5)
UDG_1_Rc = UDG_1_Rc[skill1]
UDG_1_Mc = UDG_1_Mc[skill1]
UDG_1_rh = UDG_1_rh[skill1] * 1e-3  # transform to kpc
UDG_2_Rc = UDG_2_Rc[skill2]
UDG_2_Mc = UDG_2_Mc[skill2]
UDG_2_rh = UDG_2_rh[skill2] * 1e-3
UDG_Mc = np.append(UDG_1_Mc, UDG_2_Mc)
UDG_Rc = np.append(UDG_1_Rc, UDG_2_Rc)
UDG_rh = np.append(UDG_1_rh, UDG_2_rh)

UDG_Mc = UDG_Mc[np.where(UDG_Rc <= 5)]
UDG_Rc = UDG_Rc[np.where(UDG_Rc <= 5)]
UDG_rh = UDG_rh[np.where(UDG_Rc <= 5)]
# Get three mass bin with equal width in log space from data
binwidth = np.linspace(np.log10(UDG_Mc).min(), np.log10(UDG_Mc).max(), 4)


######################## set up the environment #########################
def calculate_distance(xvlist1, xvlist2):
    d2 = xvlist1[0] ** 2 + xvlist2[0] ** 2 - 2 * xvlist1[0] * xvlist2[0] * np.cos(xvlist1[1] - xvlist2[1]) + (
            xvlist1[2] - xvlist2[2]) ** 2
    return np.sqrt(d2)


def calculate_vel(xvlist1, xvlist2):
    vx1 = xvlist1[3] * np.cos(xvlist1[1]) - xvlist1[4] * np.sin(xvlist1[1])
    vx2 = xvlist2[3] * np.cos(xvlist2[1]) - xvlist2[4] * np.sin(xvlist2[1])
    vy1 = xvlist1[3] * np.sin(xvlist1[1]) + xvlist1[4] * np.cos(xvlist1[1])
    vy2 = xvlist2[3] * np.sin(xvlist2[1]) + xvlist2[4] * np.cos(xvlist2[1])
    vrel = (vx1 - vx2) ** 2 + (vy1 - vy2) ** 2 + (xvlist1[5] - xvlist2[5]) ** 2
    return np.sqrt(vrel)


def calculate_pot(xvlist1, xvlist2, m1, m2):
    U = -cfg.G * m1 * m2 / (calculate_distance(xvlist1, xvlist2) ** 2.11 + (1.7 * 10e-3) ** 2.11) ** (1 / 2.11)
    return U


def calculate_rhalf(rhohalf1, rhohalf2, m1, m2):
    new_rhohalf = ((m1 ** (5 / 3) * rhohalf1 ** (1 / 3) + m2 ** (5 / 3) * rhohalf2 ** (1 / 3)) / (m1 + m2) ** (
                5 / 3)) ** 3
    rh = (3 * (m1 + m2) / cfg.FourPi / 2 / new_rhohalf) ** (1 / 3)

    return rh


def newxv(xvlist1, xvlist2, m1, m2):
    vx1 = xvlist1[3] * np.cos(xvlist1[1]) - xvlist1[4] * np.sin(xvlist1[1])
    vx2 = xvlist2[3] * np.cos(xvlist2[1]) - xvlist2[4] * np.sin(xvlist2[1])
    vy1 = xvlist1[3] * np.sin(xvlist1[1]) + xvlist1[4] * np.cos(xvlist1[1])
    vy2 = xvlist2[3] * np.sin(xvlist2[1]) + xvlist2[4] * np.cos(xvlist2[1])
    vz1 = xvlist1[5]
    vz2 = xvlist2[5]
    x1 = xvlist1[0] * np.cos(xvlist1[1])
    y1 = xvlist1[0] * np.sin(xvlist1[1])
    z1 = xvlist1[2]
    x2 = xvlist2[0] * np.cos(xvlist2[1])
    y2 = xvlist2[0] * np.sin(xvlist2[1])
    z2 = xvlist2[2]

    newx = (m1 * x1 + m2 * x2) / (m1 + m2)
    newy = (m1 * y1 + m2 * y2) / (m1 + m2)
    newz = (m1 * z1 + m2 * z2) / (m1 + m2)
    newvx = (m1 * vx1 + m2 * vx2) / (m1 + m2)
    newvy = (m1 * vy1 + m2 * vy2) / (m1 + m2)
    newvz = (m1 * vz1 + m2 * vz2) / (m1 + m2)

    if newx >= 0 and newy >= 0:
        newphi = np.arctan(newy / newx)
    elif newx >= 0 and newy <= 0:
        newphi = np.arctan(newy / newx) + 2 * np.pi
    elif newx <= 0 and newy >= 0:
        newphi = np.arctan(newy / newx) + np.pi
    elif newx <= 0 and newy <= 0:
        newphi = np.arctan(newy / newx) + np.pi

    return [np.sqrt(newx ** 2 + newy ** 2), newphi, newz,
            newvx * np.cos(newphi) + newvy * np.sin(newphi),
            newvy * np.cos(newphi) - newvx * np.sin(newphi),
            newvz]


def ftidal(x, ft1=0.77, eta=0.19):
    """
    Calibrated to the Penarrubia+10 tidal track, as done in
    test_evolve_StarCluster_20220717CalibrateftUsingTidalTrack.py
    """
    return ft1 * x ** eta
    # return 0.5 # <<< test: constant ft


# ---numerical resolution
cfg.Mres = 100.  # [M_sun]
cfg.Rres = 1e-2  # [kpc]

# ---dynamical friction
cfg.lnL_type = 1  # <<< Coulomb log choice, 0 = Bar+22
# ---tidal heating and evaporation efficiencies
fr = 0.08
StrippingEfficiency = 0.55  # default value (as calibrated for subhalos)
xie = 0.0074  # default is 1/137 = 0.0074 for isolated relaxed cluster




merger_r = []
merger_time = []
merger_mass = []
encounter_number=[]





#


Mb = 10 ** 8.3343  # baryon mass [M_sun]
r0 = 3.42
#r0 = 3.79178427# [kpc] # Hernquist scale radius
cb = 6.1650  # Burkert-like concentration
# ---host properties
cfg.Mh=10. ** 9.82
#cfg.Mh=10. ** 9.44359124
Mh = cfg.Mh  # halo virial mass [M_sun]
print(np.log10(Mh))
#ch = 10**0.44588193
ch = 22.05
#
h = pr.Burkert(Mh, ch, Delta=200., z=0.)
b = pr.Hernquist(Mb, r0)
Bur = pr.Burkert_like(Mb, cb)

# ---choose the potential to study
potential = [h,Bur]
profile = b

print("Energy Distribution Calculating")
aux_x = np.logspace(-1, 4, 3000)
aux_profile = h
aux_phi = -aux_profile.Phi(aux_x)
aux_drhodphi2 = h.d2rhodPhi2(aux_x)
aux_f_drhodphi2 = interp1d(aux_phi, aux_drhodphi2, fill_value=(0, 0), bounds_error=False)


def aux_f_Edd(E):
    inte = quad(lambda phi: aux_f_drhodphi2(phi) / np.sqrt(E - phi), 0, E, limit=500, epsabs=1e-5, epsrel=1e-5)[0]
    return inte / np.sqrt(8) / np.pi ** 2


Elist = []
for i in np.linspace(0.0001, -pr.Phi(aux_profile, 0.1), 1000):
    Elist.append(aux_f_Edd(i))
Elist = np.array(Elist)
cfg.f_E = interp1d(np.linspace(0.0001, -pr.Phi(aux_profile, 0.1), 1000), Elist, fill_value=(0, 0),
                   bounds_error=False)

print("Energy Distribution Calculated")

lgMmin=5.5
lgMmax=8

final_mass=[]
final_radius=[]
final_rh=[]



eta = 2.  # half of EFF outer slope

Ncluster = 300

xv0 = init.orbit_StarCluster(Ncluster, b, h, rmin=0.1, rmax=5).T
cfg.Ntot = None  # [important!] reset normalization of ICMF
m0 = init.Mass_StarCluster(N=Ncluster, alpha=1., lgMmin=lgMmin, lgMmax=lgMmax)  # [M_sun]

#xv0 = init.orbit_StarCluster_Modak2(Ncluster, h, 1.18, 0.9, rmin=1, rmax=4).T
#mt = 10 ** np.random.normal(5, np.sqrt(0.2), Ncluster)
#m0 = mt * np.exp(10 / 23)
lhalf0 = init.Reff_StarCluster_WithScatter(m0)
a0 = lhalf0 / (np.sqrt(2. ** (2. / (2 * eta - 3.)) - 1.))  # EFF scale radius

# ---for evolution and bookkeeping
Nstep = 200  # number of timesteps
tmax = 10.  # [Gyr]
timesteps = np.linspace(1e-6, tmax, Nstep)  # [Gyr]

############################### compute #################################

print('>>> initializing ... ')

clusters = [0] * Ncluster
orbits = [0] * Ncluster
HalfMassRadius = [lhalf0]
mass = [m0]
radius = [xv0.T[0]]
print('>>> evolving ... ')
t1 = time.time()
for i in range(Nstep):

    t = timesteps[i]
    if i == 0:  # for the initial step
        dt = t
    else:
        dt = t - timesteps[i - 1]

    for j in range(len(clusters)):
        if i == 0:  # initialize if at the first step
            m = m0[j]
            a = a0[j]
            clusters[j] = pr.EFF(m, a, eta)
            orbits[j] = orbit(xv0[j])
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
        #s=pr.EFF(m0[j] * np.exp(-t / 23), s.rhalf / (np.sqrt(2. ** (2. / (2 * eta - 3.)) - 1.)) , eta)

        # ---update satellite profile for the next time step
        #   Note that there is no need to update the orbit object since
        #   the coordinates are updated internally
        orbits[j] = orbit(xv)
        clusters[j] = s

    # merger
    """
    distance_matrix = np.ones((len(clusters), len(clusters)))
    energy_matrix = np.ones((len(clusters), len(clusters)))
    for m in range(len(clusters)):
        for n in range(len(clusters)):
            if m == n:
                distance_matrix[m][n] = 0
            elif calculate_distance(orbits[m].xv, orbits[n].xv) > 2 * (clusters[m].rhalf + clusters[n].rhalf):
                distance_matrix[m][n] = 0
    if len(np.where(distance_matrix == 1)[0]) != 0:
        skill1, skill2 = np.where(distance_matrix == 1)
        for l in range(len(skill1)):
            encounter_number[-1]=encounter_number[-1]+1
            m = skill1[l]
            n = skill2[l]
            if clusters[m].Mb > cfg.Mres and clusters[m].rhalf < 0.1 and clusters[n].Mb > cfg.Mres and clusters[n].rhalf < 0.1:
                mu = (clusters[m].Mb * clusters[n].Mb) / (clusters[m].Mb + clusters[n].Mb)
                energy_matrix[m][n] = round(1 / 2 * mu * calculate_vel(orbits[m].xv, orbits[n].xv) ** 2 \
                                            + calculate_pot(orbits[m].xv, orbits[n].xv, clusters[m].Mb, clusters[n].Mb))
    if len(np.where(energy_matrix < 0)[0]) != 0:
        delete_list = np.array([])
        energy_matrix_flat = energy_matrix.flatten()
        for m in np.unique(energy_matrix_flat[np.where(energy_matrix_flat < 0)]):
            skill1, skill2 = np.where(energy_matrix == m)
            if skill1[0] in delete_list or skill1[1] in delete_list:
                pass
            else:
                print(np.where(energy_matrix == m), m, t)
                merger_time[-1].append(t)
                delete_list = np.append(delete_list, skill1)
                delete_list = delete_list.astype(np.int64)
                new_mass = clusters[skill1[0]].Mb + clusters[skill1[1]].Mb
                new_lhalf = calculate_rhalf(clusters[skill1[0]].rhohalf, clusters[skill1[1]].rhohalf,
                                            clusters[skill1[0]].Mb, clusters[skill1[1]].Mb)
                orbits.append(orbit(newxv(orbits[skill1[0]].xv, orbits[skill1[1]].xv,
                                          mass[-1][skill1[0]], mass[-1][skill1[1]])))
                merger_r[-1].append(np.sqrt(orbits[-1].xv[0] ** 2 + orbits[-1].xv[2] ** 2))
                merger_mass[-1].append(new_mass)
                clusters.append(pr.EFF(new_mass, new_lhalf / (np.sqrt(2. ** (2. / (2 * eta - 3.)) - 1.)), eta))
                m0=np.append(m0,new_mass)
        orbits = list(np.delete(orbits, delete_list))
        clusters = list(np.delete(clusters, delete_list))
        m0=np.delete(m0,delete_list)
    """
    radius.append(np.array([o.xv for o in orbits]).T[0])
    mass.append(np.array([GC.Mh for GC in clusters]))
    HalfMassRadius.append(np.array([GC.rhalf for GC in clusters]))

final_mass.append(mass[-1])
final_radius.append(radius[-1])
final_rh.append(HalfMassRadius[-1])

final_mass=np.array(final_mass).flatten()
final_radius=np.array(final_radius).flatten()

# ---record


radius = np.array(radius)
mass = np.array(mass)
HalfMassRadius = np.array(HalfMassRadius)

t2 = time.time()
print('    time = %.4f sec' % (t2 - t1))

size_scatter = 10
alpha_scatter = 0.7
capsize = 5
elinewidth = 2
lw = 3

skill = np.where((mass[-1] >= 10 ** binwidth[0]) & (mass[-1] <= 10 ** binwidth[-1]) & (HalfMassRadius[-1] >= 1e-3) & (
        HalfMassRadius[-1] <= 1e-1) & (radius[-1] >= 0.1) & (radius[-1] <= 10))

model_mass0 = mass[0]
model_rh0 = HalfMassRadius[0]
model_R0 = radius[0]

model_mass = mass[-1][skill]
model_rh = HalfMassRadius[-1][skill]
model_R = radius[-1][skill]

nlevels = 7
fig, ax = plt.subplots(3, 3, figsize=(18 / 1.6, 16 / 1.6), dpi=100)
for i in range(3):
    for j in range(3):
        if i < j:
            ax[i][j].axis("off")
        elif i == j:
            ax[i][j].set_xscale('log')
            ax[i][j].tick_params('x', direction='in', top='on', right='on', length=10,
                                 width=1, which='major', labelsize=18)
            ax[i][j].tick_params('x', direction='in', top='on', right='on', length=5,
                                     width=1, which='minor', labelsize=18)
            ax[i][j].set_yticks([])
        else:
            ax[i][j].set_xscale('log')
            ax[i][j].set_yscale('log')
            ax[i][j].tick_params('both', direction='in', top='on', right='on', length=10,
                                 width=1, which='major', labelsize=18)
            ax[i][j].tick_params('both', direction='in', top='on', right='on', length=5,
                                 width=1, which='minor', labelsize=18)

ax[0][0].set_ylabel(" ")
# ax[0][0].set_xlabel("$m$ [$M_\odot$]",fontsize=18)
# ax[0][0].set_title("Density of $m$",fontsize=18)
ax[0][0].set_xticklabels([])
ax[0][0].set_yticklabels([])
# ax[0][0].set_ylim(0.05, 5)
ax[0][0].set_xlim(7e4, 2e6)

ax[1][1].set_ylabel(" ")
# ax[1][1].set_title("Density of $l_{1/2}$",fontsize=18)
# ax[1][1].set_xlabel("$l_{1/2}$ [kpc]",fontsize=18)
ax[1][1].set_xticklabels([])
ax[1][1].set_yticklabels([])
# ax[1][1].set_ylim(0.05, 5)
ax[1][1].set_xlim(1e-3, 1e-1)

ax[2][2].set_ylabel(" ")
# ax[2][2].set_title("Density of $r$",fontsize=18)
ax[2][2].set_xlabel("$R$ [kpc]", fontsize=18)
# ax[2][2].set_xticklabels([])
ax[2][2].set_yticklabels([])
# ax[2][2].set_ylim(0.05, 10)
ax[2][2].set_xlim(0.1, 10)

ax[1][0].set_ylabel(r"$l_{1/2}$ [kpc]", fontsize=18)
# ax[1][0].set_xlabel("$m$ [$M_\odot$]",fontsize=18)
ax[1][0].set_xticklabels([])
ax[1][0].set_ylim(1e-3, 1e-1)
ax[1][0].set_xlim(7e4, 2e6)

ax[2][0].set_ylabel("$R$ [kpc]", fontsize=18)
ax[2][0].set_xlabel("$m$ [$M_\odot$]", fontsize=18)
ax[2][0].set_ylim(0.1, 10)
ax[2][0].set_xlim(7e4, 2e6)

# ax[2][1].set_ylabel("$r$ [kpc]",fontsize=18)
ax[2][1].set_yticklabels([])
ax[2][1].set_xlabel(r"$l_{1/2}$ [kpc]", fontsize=18)
ax[2][1].set_ylim(0.1, 10)
ax[2][1].set_xlim(1e-3, 1e-1)

sns.kdeplot(UDG_Mc, ax=ax[0][0], color="gray", cut=0)

sns.kdeplot(UDG_Rc, ax=ax[2][2], color="gray", cut=0)

sns.kdeplot(UDG_rh, ax=ax[1][1], color="gray", cut=0)

sns.kdeplot(model_mass0, ax=ax[0][0], color="blue", cut=0)
sns.kdeplot(model_rh0, ax=ax[1][1], color="blue", cut=0)
sns.kdeplot(model_R0, ax=ax[2][2], color="blue", cut=0)

sns.kdeplot(model_mass, ax=ax[0][0], color="red", cut=0)
sns.kdeplot(model_rh, ax=ax[1][1], color="red", cut=0)
sns.kdeplot(model_R, ax=ax[2][2], color="red", cut=0)

ax[1][0].scatter(model_mass, model_rh, facecolor="red", edgecolor="k", s=size_scatter + 5, alpha=alpha_scatter - 0.3,
                 marker="s")
ax[2][0].scatter(model_mass, model_R, facecolor="red", edgecolor="k", s=size_scatter + 5, alpha=alpha_scatter - 0.3,
                 marker="s")
ax[2][1].scatter(model_rh, model_R, facecolor="red", edgecolor="k", s=size_scatter + 5, alpha=alpha_scatter - 0.3,
                 marker="s")

for i in range(1, len(binwidth)):
    skill = np.where((10 ** binwidth[i - 1] - 1 <= UDG_Mc) & (UDG_Mc <= 10 ** binwidth[i] + 1))
    ax[2][0].errorbar(x=np.median(UDG_Mc[skill]), y=np.median(UDG_Rc[skill]),
                      xerr=[[np.median(UDG_Mc[skill]) - np.percentile(UDG_Mc[skill], 16)],
                            [np.percentile(UDG_Mc[skill], 84) - np.median(UDG_Mc[skill])]],
                      yerr=[[np.median(UDG_Rc[skill]) - np.percentile(UDG_Rc[skill], 16)],
                            [np.percentile(UDG_Rc[skill], 84) - np.median(UDG_Rc[skill])]],
                      mfc="gray", mec="k", ecolor="black", capsize=capsize, elinewidth=elinewidth, ms=size_scatter,
                      alpha=alpha_scatter,
                      marker="o")
    ax[1][0].errorbar(x=np.median(UDG_Mc[skill]), y=np.median(UDG_rh[skill]),
                      xerr=[[np.median(UDG_Mc[skill]) - np.percentile(UDG_Mc[skill], 16)],
                            [np.percentile(UDG_Mc[skill], 84) - np.median(UDG_Mc[skill])]],
                      yerr=[[np.median(UDG_rh[skill]) - np.percentile(UDG_rh[skill], 16)],
                            [np.percentile(UDG_rh[skill], 84) - np.median(UDG_rh[skill])]],
                      mfc="gray", mec="k", ecolor="black", capsize=capsize, elinewidth=elinewidth, ms=size_scatter,
                      alpha=alpha_scatter,
                      marker="o")
    ax[2][1].errorbar(x=np.median(UDG_rh[skill]), y=np.median(UDG_Rc[skill]),
                      xerr=[[np.median(UDG_rh[skill]) - np.percentile(UDG_rh[skill], 16)],
                            [np.percentile(UDG_rh[skill], 84) - np.median(UDG_rh[skill])]],
                      yerr=[[np.median(UDG_Rc[skill]) - np.percentile(UDG_Rc[skill], 16)],
                            [np.percentile(UDG_Rc[skill], 84) - np.median(UDG_Rc[skill])]],
                      mfc="gray", mec="k", ecolor="black", capsize=capsize, elinewidth=elinewidth, ms=size_scatter,
                      alpha=alpha_scatter,
                      marker="o")
    ax[2][1].errorbar(x=np.median(UDG_rh[skill]), y=np.median(UDG_Rc[skill]),
                      xerr=[[np.median(UDG_rh[skill]) - np.percentile(UDG_rh[skill], 16)],
                            [np.percentile(UDG_rh[skill], 84) - np.median(UDG_rh[skill])]],
                      yerr=[[np.median(UDG_Rc[skill]) - np.percentile(UDG_Rc[skill], 16)],
                            [np.percentile(UDG_Rc[skill], 84) - np.median(UDG_Rc[skill])]],
                      mfc="white", mec="white", ecolor="black", capsize=capsize, elinewidth=elinewidth, ms=size_scatter,
                      alpha=alpha_scatter,
                      marker="$" + str(i) + "$")

    skill2 = np.where((10 ** binwidth[i - 1] - 1 <= model_mass) & (model_mass <= 10 ** binwidth[i] + 1))
    if len(skill2[0]) > 0:
        ax[2][0].errorbar(x=np.median(model_mass[skill2]), y=np.median(model_R[skill2]),
                          xerr=[[np.median(model_mass[skill2]) - np.percentile(model_mass[skill2], 16)],
                                [np.percentile(model_mass[skill2], 84) - np.median(model_mass[skill2])]],
                          yerr=[[np.median(model_R[skill2]) - np.percentile(model_R[skill2], 16)],
                                [np.percentile(model_R[skill2], 84) - np.median(model_R[skill2])]],
                          mfc="red", mec="k", ecolor="darkred", capsize=capsize, elinewidth=elinewidth, ms=size_scatter,
                          alpha=alpha_scatter,
                          marker="o")
        ax[1][0].errorbar(x=np.median(model_mass[skill2]), y=np.median(model_rh[skill2]),
                          xerr=[[np.median(model_mass[skill2]) - np.percentile(model_mass[skill2], 16)],
                                [np.percentile(model_mass[skill2], 84) - np.median(model_mass[skill2])]],
                          yerr=[[np.median(model_rh[skill2]) - np.percentile(model_rh[skill2], 16)],
                                [np.percentile(model_rh[skill2], 84) - np.median(model_rh[skill2])]],
                          mfc="red", mec="k", ecolor="darkred", capsize=capsize, elinewidth=elinewidth, ms=size_scatter,
                          alpha=alpha_scatter,
                          marker="o")
        ax[2][1].errorbar(x=np.median(model_rh[skill2]), y=np.median(model_R[skill2]),
                          xerr=[[np.median(model_rh[skill2]) - np.percentile(model_rh[skill2], 16)],
                                [np.percentile(model_rh[skill2], 84) - np.median(model_rh[skill2])]],
                          yerr=[[np.median(model_R[skill2]) - np.percentile(model_R[skill2], 16)],
                                [np.percentile(model_R[skill2], 84) - np.median(model_R[skill2])]],
                          mfc="red", mec="k", ecolor="darkred", capsize=capsize, elinewidth=elinewidth, ms=size_scatter,
                          alpha=alpha_scatter,
                          marker="o")
        ax[2][1].errorbar(x=np.median(model_rh[skill2]), y=np.median(model_R[skill2]),
                          xerr=[[np.median(model_rh[skill2]) - np.percentile(model_rh[skill2], 16)],
                                [np.percentile(model_rh[skill2], 84) - np.median(model_rh[skill2])]],
                          yerr=[[np.median(model_R[skill2]) - np.percentile(model_R[skill2], 16)],
                                [np.percentile(model_R[skill2], 84) - np.median(model_R[skill2])]],
                          mfc="white", mec="white", ecolor="darkred", capsize=capsize, elinewidth=elinewidth,
                          ms=size_scatter, alpha=alpha_scatter,
                          marker="$" + str(i) + "$")

ax[1][0].scatter(UDG_Mc, UDG_rh, facecolor="gray", edgecolor="k", s=size_scatter + 5, alpha=alpha_scatter - 0.3,
                 marker="s")
ax[2][0].scatter(UDG_Mc, UDG_Rc, facecolor="gray", edgecolor="k", s=size_scatter + 5, alpha=alpha_scatter - 0.3,
                 marker="s")
ax[2][1].scatter(UDG_rh, UDG_Rc, facecolor="gray", edgecolor="k", s=size_scatter + 5, alpha=alpha_scatter - 0.3,
                 marker="s")
for i in range(3):
    for j in range(3):
        if i > j:
            for k in binwidth:
                ax[i][j].axvline(x=10 ** k, lw=lw, color="darkgreen", linestyle="--")

ax[0][2].text(x=0.05, y=0.75, s="Model: initial", transform=ax[0][2].transAxes, color="blue", fontsize=21)
ax[0][2].text(x=0.05, y=0.5, s="Model: evolved", transform=ax[0][2].transAxes, color="red", fontsize=21)
ax[0][2].text(x=0.05, y=0.25, s="Data", transform=ax[0][2].transAxes, color="gray", fontsize=21)

plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.show()
