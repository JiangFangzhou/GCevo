#Example code to calculate energy distribution for Eddington Inversion
#One can use different profiles and their analytics to calculate energy distribution



import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

import GCevo.config as cfg
import GCevo.profiles as pr




h=pr.NFW(1e12,10)

print("Energy Distribution Calculating")
E_x = np.logspace(-1, 4, 3000) #Define the radius range for distribution
E_profile = h
E_phi = -E_profile.Phi(E_x) #Calculate the second derivative of energy relative to potential and do tabulation
E_drhodphi2 = h.d2rhodPhi2(E_x)
E_f_drhodphi2 = interp1d(E_phi, E_drhodphi2, fill_value=(0, 0), bounds_error=False)

#define energy distribution function
def E_f_Edd(E):
    inte = quad(lambda phi: E_f_drhodphi2(phi) / np.sqrt(E - phi), 0, E, limit=500, epsabs=1e-5, epsrel=1e-5)[0]
    return inte / np.sqrt(8) / np.pi ** 2

#do tabulation to get the energy distribution as function of energy
Elist = []
for i in np.linspace(0.0001, -pr.Phi(E_profile, 0.1), 1000):
    Elist.append(E_f_Edd(i))
Elist = np.array(Elist)
cfg.f_E = interp1d(np.linspace(0.0001, -pr.Phi(E_profile, 0.1), 1000), Elist, fill_value=(0, 0),
                   bounds_error=False)

print("Energy Distribution Calculated")