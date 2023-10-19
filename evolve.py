################## Functions for satellite evolution ####################

# Arthur Fangzhou Jiang 2022, Caltech, Carnegie
# Jinning Liang 2022 Wuhan University

#########################################################################
import sys

import GCevo.config as cfg
import GCevo.profiles as pr

import numpy as np
from scipy.optimize import brentq

#########################################################################

# ---star-cluster evolution

def EFF(sp, potential, dt, xv, ft=3., fr=0.1, xie=0.0074,
        choice='King62', alpha=1.):
    """
    Evolve a star cluster described by an EFF87 profile for one timestep
    considering 
    (1) mass loss due to tidal striping from the host potential
    
        dm_t / m = - alpha * [m - m(l_t)]/m * dt/t_dyn 
                := - alpha * xi_t * dt / t_dyn
        
        where 
        
        dm_t is the mass change (negative) due to tidal stripping
        m is the satellite mass; 
        m(l_t) is the satellite mass within the tidal radius l_t; 
        dt is the time step; 
        t_dyn is the host dynamical time at the satellite's position;
        xi_t := [m - m(l_t)]/m is the fraction of stellar mass outside 
            the tidal radius 
        
    (2) mass loss due to evaporation after two-body relaxation
    
        dm_r / m = - xi_e * dt / t_r
        
        where
        
        dm_r is the mass change (negative) due to evaporation
        m is the satellite mass; 
        xi_e is the fraction of stars in the high-velocity tail of the
            Maxwellian distribution with v > v_escape
        dt is the time step; 
        t_r is the cluster's relaxation time.
    
    Syntax:
    
        EFF(sp,potential,dt,xv,ft=3.,fr=0.1,xie=0.0074, 
            choice='King62',alpha=1.)
    
    where
        
        sp: star cluster potential (an object of the EFF class as defined
            in profiles.py)
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        dt: time interval [Gyr] (float)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
        ft: tidal heating efficiency (float, default=3., appropriate for 
            star clusters with an outer density slope of -4, e.g., EFF)
        fr: evaporation efficiency (float, default=0.1)
        choice: choice of tidal radius expression, including
            "King62" (default): eq.(12.21) of Mo, van den Bosch, White 10
            "Tormen98": eq.(3) of van den Bosch+17
        alpha: tidal stripping calibration parameter (default=1.)
        
    Return:
        
        the evolved EFF class object, 
        tidal radius [kpc]
    """
    # ---mass loss from tidal stripping by the host potential
    lt = ltidal(sp, potential, xv, choice)
    xit = (sp.Mh - sp.M(lt)) / sp.Mh
    xit = max(xit, 1e-6)  # <<< safety: avoid negative or zero xi_t
    if lt > 10 * sp.rhalf:
        xit = 1e-6
    tdyn = pr.tdyn(potential, xv[0], xv[2])
    dmt = - alpha * xit * sp.Mh * dt / tdyn
    # ---mass loss from evaporation
    tr = tau_relax_cluster(sp.Mh, sp.rhohalf)
    dmr = - xie * dt / tr
    # ---update mass
    m = max(sp.Mh + dmt + dmr, cfg.Mres)
    # ---update structure
    fac = alpha * (5. - 3. / ft) * xit + xie * (5. - 3. / fr) * tdyn / tr
    drhohalf = fac * dt / tdyn * sp.rhohalf
    rhohalf = sp.rhohalf + drhohalf
    if rhohalf > 0.:
        lhalf = (3. * m / (8. * np.pi * rhohalf)) ** (1. / 3.)
    else:  # safety: if density drops below zero, it means the cluster
        # is disrupted, we set the radius to be arbitrarily large
        lhalf = cfg.Rres
    eta = sp.eta
    a = lhalf / (np.sqrt(2. ** (2. / (2 * eta - 3.)) - 1.))
    return pr.EFF(m, a, eta), lt


def tau_relax_cluster(m, rhohalf):
    """
    Timescale [Gyr] for the internal two-body relaxation of star clusters 
    following Gieles & Renaud (2016):
    
        tau_rlx = kappa (m/10^4 M_sun) /sqrt(rho_half/10^11 M_sun kpc^-3)
        
    where
    
        kappa ~ 0.142 Gyr (Spitzer & Hart 71)
    
    Syntax:
    
        tau_relax_GR16(m,rhohalf)
        
    where
    
        m: mass of star cluster [M_sun]
        rhohalf: average density within half-mass radius [M_sun/kpc^3] 
            (float)
    """
    return 0.142 * (m / 1e4) / np.sqrt(rhohalf / 1e11)


# ---tidal stripping



def msub(sp, potential, xv, dt, choice='King62', alpha=1.):
    """
    Evolve subhalo mass due to tidal stripping, by an amount of
    
        alpha * [m - m(l_t)] * dt/t_dyn
        
    where 
    
        m is the satellite virial mass; 
        m(l_t) is the satellite mass within the tidal radius l_t; 
        dt is the time step; 
        t_dyn is the host dynamical locally at the satellite's position.
    
    Syntax:
    
        msub(sp,potential,xv,dt,choice='King62',alpha=1.)
    
    where
        
        sp: satellite potential (an object of one of the classes defined 
            in profiles.py)
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
        dt: time interval [Gyr] (float)
        choice: choice of tidal radius expression, including
            "King62" (default): eq.(12.21) of Mo, van den Bosch, White 10
            "Tormen98": eq.(3) of van den Bosch+17
        alpha: stripping efficienty parameter -- the larger the 
            more effient (default=1.)
            
    Return
    
        evolved mass, m [M_sun] (float)
        tidal radius, l_t [kpc] (float)
    """
    lt = ltidal(sp, potential, xv, choice)
    if lt < sp.rh:
        dm = alpha * (sp.Mh - sp.M(lt)) * dt / pr.tdyn(potential, xv[0], xv[2])
        dm = max(dm, 0.)  # avoid negative dm
        if cfg.Mres is not None:
            # Fixed Mres case
            m = max(sp.Mh - dm, cfg.Mres)
        else:
            # Evolve subhaloes down to m/m_{acc} = phi_{res}
            m = max(sp.Mh - dm, cfg.phi_res * sp.Minit)
    else:
        m = sp.Mh
    return m, lt


def ltidal(sp, potential, xv, choice='King62'):
    """
    Tidal radius [kpc] of a satellite, given satellite profile, host
    potential, and phase-space coordinate within the host. 
    
    Syntax:
    
        ltidal(sp,potential,xv,choice='King62')
    
    where
    
        sp: satellite potential (an object define in profiles.py)
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
        choice: choice of tidal radius expression, including
            "King62" (default): eq.(12.21) of Mo, van den Bosch, White 10
            "Tormen98": eq.(3) of van den Bosch+18
            
    Note that the only difference between King62 and Tormen98 is that 
    the latter ignores the centrifugal force and thus gives larger tidal 
    radius (i.e., weaker tidal stripping). 
    """
    a = cfg.Rres
    b = 9.999 * sp.rh  # <<< ??? Sheridan put this? Why 9.999?
    if choice == 'King62':
        rhs = lt_King62_RHS(potential, xv)
    elif choice == 'Tormen98':
        rhs = lt_Tormen98_RHS(potential, xv)
    else:
        sys.exit('Invalid choice of tidal radius type!')

    fa = Findlt(a, sp, rhs)
    fb = Findlt(b, sp, rhs)
    if fa * fb > 0.:
        #lt = cfg.Rres
        lt = sp.rh
    else:
        lt = brentq(Findlt, a, b, args=(sp, rhs),
                    rtol=1e-5, maxiter=1000)
    return lt


def lt_Tormen98_RHS(potential, xv):
    """
    Auxiliary function for 'ltidal', which returns the right-hand side
    of the Tormen98 equation for tidal radius, as in eq.(3) of 
    van den Bosch+18, but inverted and with all subhalo terms on
    left-hand side.
    
    Syntax:
    
        lt_Tormen98_RHS(potential,xv)
    
    where
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
    """
    r = np.sqrt(xv[0] ** 2. + xv[2] ** 2.)
    M = pr.M(potential, r)
    rho = pr.rho(potential, r)
    dlnMdlnr = cfg.FourPi * r ** 3 * rho / M
    return (M / r ** 3) * (2. - dlnMdlnr)


def lt_King62_RHS(potential, xv):
    """
    Auxiliary function for 'ltidal', which returns the right-hand side
    of the King62 equation for tidal radius, as in eq.(12.21) of 
    Mo, van den Bosch, White 10, but inverted and with all subhalo
    terms on left-hand side.
    
    Syntax:
    
        lt_King62_RHS(potential,xv)
    
    where
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
    """
    r = np.sqrt(xv[0] ** 2. + xv[2] ** 2.)
    Om = Omega(xv)
    M = pr.M(potential, r)
    rho = pr.rho(potential, r)
    dlnMdlnr = cfg.FourPi * r ** 3 * rho / M
    return (M / r ** 3) * (2. + Om ** 2. * r ** 3 / cfg.G / M - dlnMdlnr)


def Findlt(l, sp, rhs):
    """
    Auxiliary function for 'ltidal', which returns the 
    
        left-hand side - right-hand side
    
    of the equation for tidal radius. Note that this works
    for either the Tormen98 or King62, since all differences
    are contained in the pre-computed right-hand side.
    
    Syntax:
    
        Findlt(l,sp,rhs)
    
    where
    
        l: radius in the satellite [kpc] (float)
        sp: satellite potential (an object define in profiles.py)
        rhs: right-hand side of equation, computed by either
        lt_Tormen98_RHS() or lt_King62_RHS() (float)
    """
    m = sp.M(l)
    return (m / l ** 3) - rhs


def Omega(xv):
    """
    Angular speed [Gyr^-1] upon input of phase-space coordinates
    
    Syntax: 
    
        Omega(xv):
    
    where
    
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
    """
    rsqr = xv[0] ** 2. + xv[2] ** 2.
    rxv = np.cross(np.array([xv[0], 0., xv[2]]), xv[3:6])
    return np.sqrt(rxv[0] ** 2. + rxv[1] ** 2. + rxv[2] ** 2.) / rsqr

