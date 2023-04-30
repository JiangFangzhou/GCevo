#########################################################################
#
# global variables 
# 
# import config as cfg in all related modules, and use a global variable
# x defined here in the other modules as cfg.x 
#
#
# Arthur Fangzhou Jiang 2022 Caltech & Carnegie
# Jinning Liang 2022 Wuhan University

#########################################################################

import numpy as np

########################## user control #################################
#---cosmology
h = 0.7
Om = 0.3
Ob = 0.0465
OL = 0.7
s8 = 0.8
ns = 1.


Mres = None #[solar mass] mass resolution

phi_res = 10**-5 # Resolution in m/m_{acc}
Rres = 0.001 # [kpc] spatial resolution
lnL_pref = 0.75 # multiplier for Coulomb log (fiducial 0.75) for Petts+16
lnL_const = 3. # constant Coulomb log value for 
# NOTE: The lnL_pref default is 0.75, calibrated in Green+20
# A typical default would be lnL_pref = 1.0
lnL_type = 0 # indicates using log(Mh/Ms) (instantaneous)
lnL_type = 1 # indicates using the Petts+15 Couloumb logarithm
lnL_type = 2 # indicates using constant Couloumb logarithm of lnL_const
lnL_type = 3 # indicates using the Bar+21/22 Couloumb logarithm
lnL_type = 4 # indicates using the Modak+22 Couloumb logarithm
bmax=0.5 # constant maximal impact parameter value for Bar+21/22
Ntot = None # the normalization of the initial cluster mass function
#pre-calculate velocity dispersion to speed up
logicI=True
I0=0
#Halo mass
Mh=1e11
#Find Energy distribution by Eddington Inversion
f_E=None

############################# constants #################################

G = 4.4985e-06 # gravitational constant [kpc^3 Gyr^-2 Msun^-1]
rhoc0 = 277.5 # [h^2 Msun kpc^-3]
ln10 = np.log(10.)
Root2 = np.sqrt(2.)
RootPi = np.sqrt(np.pi)
Root2OverPi = np.sqrt(2./np.pi)
Root1Over2Pi = np.sqrt(0.5/np.pi)
TwoOverRootPi = 2./np.sqrt(np.pi)
FourOverRootPi = 4./np.sqrt(np.pi)
FourPiOverThree = 4.*np.pi/3.
TwoPi = 2.*np.pi
TwoPiG = 2.*np.pi*G
TwoPisqr = 2.*np.pi**2
ThreePi = 3.*np.pi
FourPi = 4.*np.pi
FourPiG = 4.*np.pi*G
FourPiGsqr = 4.*np.pi * G**2. # useful for dynamical friction 
ThreePiOverSixteenG = 3.*np.pi / (16.*G) # useful for dynamical time
kms2kpcGyr = 1.0227 # multiplier for converting velocity from [km/s] to 
    #[kpc/Gyr] <<< maybe useless, as we may only work with kpc and Gyr
eps = 0.001 # an infinitesimal for various purposes: e.g., if the 
    # fractional difference of a quantify between two consecutive steps
    # is smaller than cfg.eps, jump out of a loop; and e.g., for 
    # computing derivatives

