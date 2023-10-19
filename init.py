############### Functions for initializing satellites ###################

# Arthur Fangzhou Jiang 2022, Caltech, Carnegie
# Jinning Liang 2022 Wuhan University

#########################################################################

import numpy as np

import GCevo.config as cfg
import GCevo.profiles as pr

from scipy.stats import lognorm, expon, rv_continuous
from scipy.integrate import quad
from scipy.interpolate import interp1d


#########################################################################


# ---for initial star-cluster properties

def Reff_StarCluster(M, R4=0.00255, beta=0.24):
    """
    Size [kpc] - mass [M_sun] relation of (young) star clusters based on
    the Legacy Extragalactic UV Survey (LEGUS) result presented in
    Brown & Gnedin (2021).

    Syntax:

        Reff_StarCluster(M,R4=0.00255,beta=0.24)

    where

        M: mass of zero-age star cluster [M_sun] (float or array)
        R4: normalization [kpc] (float, default=0.00255)
        beta: slope (float, default=0.24)

    Note that the default values of R4 and beta correspond to the
    relation for young star clusters. As the clusters evolve, these
    values may change.

    Return:

        Half-mass radius [kpc] (float or array)
    """
    return R4 * (M / 1e4) ** beta


def Reff_StarCluster_WithScatter(M, R4=0.00255, beta=0.24):
    """
    Same as Reff_StarCluster, but with a log-normal scatter
    """
    Reff_median = Reff_StarCluster(M, R4, beta)
    mu = np.log10(Reff_median)
    return 10. ** np.random.normal(mu, 0.2)


def dNdlgM_StarCluster_TG19(lgM, alpha=1., lgMmin=4., lgMmax=7.5):
    """
    Un-normalized initial star clusters mass function of 
    Trujillo-Gomez et al. (2019), which takes the form of a 
    double-truncated Schechter function
     
        dN / dlnM ~ M^-alpha exp(-M_min/M) exp(-M/M_max)
        
    where 
        
        alpha: negative of power law slope of dN/dln(M)
        M_min: low-mass truncation scale
        M_max: high-mass truncation scale
    
    Syntax:
    
        dNdlgM_StarCluster_TG19(lgM,A=1.,alpha=1.,lgMmin=4.,lgMmax=7.5)
        
    where
    
        lgM: log mass [M_sun]  of zero-age star cluster (float or array)
        alpha: negative power law slope of dN/dln(M) (float, default=1.)
        lgMmin: log lower characteristic mass [M_sun] (float, default=4.)
        lgMmax: log upper characteristic mass [M_sun](float, default=7.5)
        
    Return: 
    
        dN / dlog(M), where log is 10-based log.
    """
    M = 10. ** lgM
    Mmin = 10. ** lgMmin
    Mmax = 10. ** lgMmax
    return cfg.ln10 * M ** (-alpha) * np.exp(-Mmin / M) * np.exp(-M / Mmax)


def dPdlgM_StarCluster_TG19(lgM, alpha=1., lgMmin=4., lgMmax=7.5):
    """
    Normalized ICMF of Trujillo-Gomez et al. (2019),
    which serves as the PDF for drawing star cluster mass.
    
    Syntax:
    
        dPdlgM_StarCluster_TG19(lgM,alpha=1.,lgMmin=4.,lgMmax=7.5)
        
    where
    
        lgM: log mass [M_sun]  of zero-age star cluster (float or array)
        alpha: negative power law slope of dN/dln(M) (float, default=1.)
        lgMmin: log lower characteristic mass [M_sun] (float, default=4.)
        lgMmax: log upper characteristic mass [M_sun](float, default=7.5)
    
    Return: 
    
        dP / dlog(M), where log is 10-based log.
    """
    if cfg.Ntot is None:
        cfg.Ntot = quad(dNdlgM_StarCluster_TG19, 0., 10.,
                        args=(alpha, lgMmin, lgMmax),
                        epsabs=1e-7, epsrel=1 - 6, limit=10000)[0]
    else:
        pass
    return dNdlgM_StarCluster_TG19(lgM, alpha, lgMmin, lgMmax) / cfg.Ntot


def DrawMass_StarCluster(N, alpha=None, lgMmin=None, lgMmax=None):
    """
    Draw masses for N star clusters randomly according to the initial 
    star clusters mass function of Trujillo-Gomez et al. (2019).
    
    Syntax:
    
        DrawMass_StarCluster(N,A=None,alpha=None,lgMmin=None,lgMmax=None)
        
    where
    
        N: number of star clusters(float or array)
        alpha: negative power law slope of dN/dln(M) (float,default=None)
        lgMmin: log lower characteristic mass [M_sun](float,default=None)
        lgMmax: log upper characteristic mass [M_sun](float,default=None)
    
    Return:
    
        mass [M_sun] （array of length N)
    """

    class lgMassDistribution(rv_continuous):
        def _pdf(self, lgM, ):
            return dPdlgM_StarCluster_TG19(lgM, alpha, lgMmin, lgMmax)

    lgMassSampler = lgMassDistribution(a=0., b=10.)
    lgM = lgMassSampler.rvs(size=N)
    return 10. ** lgM


def Mass_StarCluster(N, alpha=None, lgMmin=None, lgMmax=None):
    """
    Draw mass for a star cluster randomly according to the initial 
    star clusters mass function of Trujillo-Gomez et al. (2019).
    
    Syntax:
    
        Mass_StarCluster(alpha=None,lgMmin=None,lgMmax=None)
        
    where
        
        alpha: negative power law slope of dN/dln(M) (float,default=None)
        lgMmin: log lower characteristic mass [M_sun](float,default=None)
        lgMmax: log upper characteristic mass [M_sun](float,default=None)
    
    Return:
    
        mass [M_sun] （array of length N)
    """
    return DrawMass_StarCluster(N, alpha, lgMmin, lgMmax)


# ---for initializing orbit

def orbit_StarCluster(N, profile, potential, rmin=1., rmax=100.):
    """
    Initialize the orbit of a satellite star cluster, given the profile 
    from which the star cluster's position is drawn, and total potential 
    in which the star cluster evolves. 
    
    We assume that the star clusters is in equilibrium with the 
    underlying potential, so the velocity can be simply the local 
    velocity dispersion multiplied by sqrt(3)
    
    We assume isotropic distribution of position vector and velocity 
    vector, i.e., the polar angle of the position vector, theta, is
    given by 
    
        arccos(2 R - 1)
    
    where R is a uniform random number between 0 and 1; and the angle
    between the position vector and the velocity vector, gamma, is also
    given by this.
    
    Syntax:

        orbit_StarCluster(profile,potential,rmin=1.,rmax=100.)

    where

        profile: the underlying profile from which we assign the star
            cluster's initial position (a density profile object as 
            defined in profiles.py, or a list of such objects)
        potential: total potential in which we evolve the star cluster
            (a density profile as defined in profiles.py or a list of 
            such objects in the case of a composite host potential)
        rmin: minimum radius [kpc] (float, default=1.)
        rmax: maximum radius [kpc] (float, default=100.)
        
    Return:

        phase-space coordinates [R,phi,z,VR,Vphi,Vz] in cylindrical frame
        [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (array)
    """
    r0 = DrawRadius(N, profile, rmin=rmin, rmax=rmax)  # draw radii from profile
    # np.random.seed(3)
    theta = np.arccos(2. * np.random.random(N) - 1.)  # i.e., isotropy
    # np.random.seed(4)
    zeta = 2. * np.pi * np.random.random(N)  # i.e., uniform azimuthal
    # angle, zeta, of velocity vector in theta-phi-r frame
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    sinzeta = np.sin(zeta)
    coszeta = np.cos(zeta)
    #V0 = np.sqrt(3.) * pr.sigma(potential, r0)
    V0=[]
    for i in r0:
        V0.append(DrawVelocity_Modak23_NFW(1, potential, i)[0])
    V0=np.array(V0)
    # np.random.seed(5)
    gamma = np.arccos(2. * np.random.random(N) - 1.)  # i.e., isotropy
    singamma = np.sin(gamma)
    cosgamma = np.cos(gamma)
    # np.random.seed(6)
    return np.array([
        r0 * sintheta,
        np.random.random(N) * 2. * np.pi,  # uniformly random phi in (0,2pi)
        r0 * costheta,
        V0 * (singamma * coszeta * costheta + cosgamma * sintheta),
        V0 * singamma * sinzeta,
        V0 * (cosgamma * costheta - singamma * coszeta * sintheta),
    ])


def DrawRadius(N, profile, rmin=1., rmax=100.):
    """
    Draw radii for N particles randomly according to a density profile.
    
    Syntax:
    
        DrawRadius(N,profile,rmin=1.,rmax=100.)
        
    where
    
        N: particle number (float or array)
        profile: density profile from which we draw the particle's radius
            (profile class defined in profiles.py, or a list of such 
            objects)
        rmin: minimum radius [kpc] (float, default=1.)
        rmax: maximum radius [kpc] (float, default=100.)
        
    Return:
    
        radii [kpc] （array of length N)
    """

    class RadiusDistribution(rv_continuous):
        def _pdf(self, r, C):
            # C is normalization constant
            r1 = r * (1. + cfg.eps)
            r2 = r * (1. - cfg.eps)
            dMdr = (pr.M(profile, r1) - pr.M(profile, r2)) / (r1 - r2)
            return (1. / C) * dMdr

    RadiusSampler = RadiusDistribution(a=rmin, b=rmax)
    C = pr.M(profile, rmax) - pr.M(profile, rmin)

    return RadiusSampler.rvs(C=C, size=N)


def DrawRadius_Modak23(N, n, Reff, rmin=0.1, rmax=10.):
    pn = 1 - 1.188 / 2 / n + 0.22 / 4 / n ** 2
    bn = 2 * n - 1 / 3 + 0.009876 / n
    C = quad(lambda r: cfg.FourPi * r ** 2 * (r / Reff) ** pn * np.exp(-bn * (r / Reff) ** (1 / n)), rmin,
             rmax)[0]

    class RadiusDistribution(rv_continuous):
        def _pdf(self, r, C, Reff):
            # C is normalization constant
            nu = (r / Reff) ** pn * np.exp(-bn * (r / Reff) ** (1 / n))
            P = cfg.FourPi * r ** 2 * nu

            return (1. / C) * P

    RadiusSampler = RadiusDistribution(a=rmin, b=rmax)

    return RadiusSampler.rvs(C=C, Reff=Reff, size=N)


def DrawVelocity_Modak23_NFW(N, profile, r, vmin=0.1, vmax=None):
    """
    x = np.logspace(-1, 3, 1000)
    phi = -pr.Phi(profile, x)
    drhodphi2 = 2 * profile.rho0 * profile.rs ** 3 * (x * (6 * x ** 2 + 4 * x * profile.rs + profile.rs ** 2) - (
                1 - cfg.TwoPi * x ** 3 * profile.rho(x) / profile.M(x)) * x * (x + profile.rs) * (
                                                                  3 * x + profile.rs)) / cfg.G ** 2 / profile.M(
        x) ** 2 / (x + profile.rs) ** 4
    f_drhodphi2=interp1d(phi, drhodphi2, fill_value="extrapolate")

    def f_Edd(E):
        inte = quad(lambda phi: f_drhodphi2(phi)/np.sqrt(E - phi), 0, E,limit=100,epsabs=1e-5,epsrel=1e-5)[0]
        return inte / np.sqrt(8) / np.pi ** 2

    Elist = []
    for i in np.linspace(0.0001, -pr.Phi(profile,r), 1000):
        Elist.append(f_Edd(i))
    Elist = np.array(Elist)
    Elist[np.where(Elist < 0)] = 0
    f_E = interp1d(np.linspace(0.0001, -pr.Phi(profile,r), 1000), Elist, fill_value=0)
    """
    if vmax==None:
        vmax=np.sqrt(2*cfg.G*pr.M(profile,r)/r)
    else:
        vmax=vmax


    C = quad(lambda v: cfg.FourPi * v ** 2 * cfg.f_E(-pr.Phi(profile, r) - v ** 2 / 2) / pr.rho(profile, r), vmin,
             vmax, limit=200, epsabs=1e-5, epsrel=1e-5)[0]


    class RadiusDistribution(rv_continuous):
        def _pdf(self, v, r):
            # C is normalization constant
            P = cfg.FourPi * v ** 2 * cfg.f_E(-pr.Phi(profile, r) - v ** 2 / 2) / pr.rho(profile, r)

            return P/C

    RadiusSampler = RadiusDistribution(a=vmin, b=vmax)

    return RadiusSampler.rvs(r=r, size=N)


def orbit_StarCluster_Modak(N, potential, n, Reff, rmin=1., rmax=100.):
    r0 = DrawRadius_Modak23(N, n, Reff, rmin=rmin, rmax=rmax)
    V0 = []
    for i in r0:
        V0.append(DrawVelocity_Modak23_NFW(1, potential, i)[0])
    V0 = np.array(V0)
    theta = np.pi * np.random.random(N)
    phi = 2. * np.pi * np.random.random(N)
    theta_v = np.pi * np.random.random(N)
    phi_v = 2. * np.pi * np.random.random(N)

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    sintheta_v = np.sin(theta_v)
    costheta_v = np.cos(theta_v)
    sinphi_v = np.sin(phi_v)
    cosphi_v = np.cos(phi_v)

    return np.array([
        r0 * sintheta,
        phi,
        r0 * costheta,
        V0 * (sintheta_v * cosphi_v * sintheta - costheta_v * costheta),
        V0 * sintheta_v * sinphi_v,
        V0 * (sintheta_v * cosphi_v * costheta + costheta_v * sintheta),
    ])


def orbit_StarCluster_Modak2(N, potential, n, Reff, rmin=1., rmax=100.):
    r0 = DrawRadius_Modak23(N, n, Reff, rmin=rmin, rmax=rmax)
    V0 = []
    for i in r0:
        V0.append(DrawVelocity_Modak23_NFW(1, potential, i)[0])
    V0 = np.array(V0)
    # np.random.seed(3)
    theta = np.arccos(2. * np.random.random(N) - 1.)  # i.e., isotropy
    # np.random.seed(4)
    zeta = 2. * np.pi * np.random.random(N)  # i.e., uniform azimuthal
    # angle, zeta, of velocity vector in theta-phi-r frame
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    sinzeta = np.sin(zeta)
    coszeta = np.cos(zeta)
    # np.random.seed(5)
    gamma = np.arccos(2. * np.random.random(N) - 1.)  # i.e., isotropy
    singamma = np.sin(gamma)
    cosgamma = np.cos(gamma)
    # np.random.seed(6)
    return np.array([
        r0 * sintheta,
        np.random.random(N) * 2. * np.pi,  # uniformly random phi in (0,2pi)
        r0 * costheta,
        V0 * (singamma * coszeta * costheta + cosgamma * sintheta),
        V0 * singamma * sinzeta,
        V0 * (cosgamma * costheta - singamma * coszeta * sintheta),
    ])





