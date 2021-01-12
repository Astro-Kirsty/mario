import numpy as np
from astropy import units as u
from spectral_cube import SpectralCube
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


@u.quantity_input(n=u.cm**(-3))
def calc_Bmag(n):
    """
    Calculation of magnetic field from Crutcher et al. 2010 

    Parameters
    ----------------------
    n : float
        The number density of the cloud
    """
    B0 = 10*u.uG
    n0 = 300*u.cm**(-3)
    
    if n < n0:
        B_mag = B0
    else:
        B_mag = B0 * (n / n0) ** 0.65
        
    return B_mag


@u.quantity_input(E_SN=u.erg, n0=u.cm**(-3), t=u.yr)
def calc_shock_radius(t, E_SN=1e51*u.erg, n0=1*u.cm**-3):
    """
    Calculates the radius of the shock of a supernova remnant in the Sedov-Taylor phase.
    It can be used as the release point of cosmic-ray protons from the accelerator, 
    if t corresponds to the escape time. (Reynolds 2008 or Truelove & McKee 1999)
    Parameters
    ----------------------
    E_SN : float
        Energy of explosion (ejected energy)
    n0 : float
        Ambient upstream hydrogen number density
    t : float
        Time after explosion for which the shock radius is to be calculated
    Returns
    ----------------------
    float
        The shock radius in the Sedov-Taylor phase
    """
    u1 = 1.4 * u.kg
    R = (0.31 * (E_SN / (1e51*u.erg)) ** (1/5) * (n0 / u.cm**(-3))** (-1/5) *
         (u1 / (1.4 * u.kg)) ** (-1 / 5) * (t / u.yr) ** (2 / 5)) * u.pc

    return R.to(u.pc)

@u.quantity_input(Emax_t_ST0=u.TeV, Emax_t=u.TeV, t=u.yr, t_ST0=u.yr)
def calc_energy_evolution(t=50*u.kyr, t_ST0=100*u.yr, Emax_t=1*u.GeV, Emax_t_ST0=1*u.PeV):
    """
    Calculates the energy evolution.
    This parameter is used to calculate the time after supernova explosion
    at which CRs with energy E are released in the ISM  (eg Gabici et al. 2009)
    Parameters
    ----------------------
    t : float
        Time after supernova explosion at which particles 
        of maximum energy Emax_t are accelerated
    t_ST0 : float
    	Start of Sedov-Taylor phase
    Emax_t : float
        Maximum energy of particles at time t after supernova explosion
    Emax_t_ST0 : float
        Maximum energy of particles at 100 yr (default) after supernova explosion
        (corresponds to start of Sedov-Taylor phase)
    Returns
    ----------------------
    float
        The energy evolution parameter 
        (often referred to as 'deltap')
    """
    return np.log(Emax_t / Emax_t_ST0) / np.log(t_ST0 / t)

@u.quantity_input(E=u.TeV, Emax_t_ST0=u.TeV, t_ST0=100*u.yr)
def calc_escape_time(E, deltap, Emax_t_ST0=1*u.PeV, t_ST0=100*u.yr):
    """
    Calculates the time after supernova explosion at which CRs
    with energy E are released in the ISM   (eg Gabici et al. 2009)
    Parameters
    ----------------------
    E: float
        The energy of CRs that are released in the ISM
    deltap : float
        Energy evolution parameter
    Emax_t_ST0 : float
        Maximum energy of particles at t_ST0 after supernova explosion
        (corresponds to start of Sedov-Taylor phase)
    t_ST0 : float
    	Start of the Sedov-Taylor phase
    Returns
    ----------------------
    float
        The time after supernova explosion at which CRs are released
    """
    tesc = t_ST0 * (E / Emax_t_ST0) ** (- 1 / deltap)
    return tesc.to(u.kyr)

@u.quantity_input(t=u.yr, Emax_t_ST0=u.TeV, t_ST0=u.yr)
def calc_escape_energy(t, deltap, Emax_t_ST0=1*u.PeV, t_ST0=100*u.yr):
    """
    Calculates the minimum energy at which CRs
    are released in the ISM at time t   (eg Gabici et al. 2009)
    Parameters
    ----------------------
    t: float
        The time at which CRs are released in the ISM
    deltap : float
        Energy evolution parameter
    Emax_t_ST0 : float
        Maximum energy of particles at t_ST0 after supernova explosion
        (corresponds to start of Sedov-Taylor phase)
    t_ST0 : float
    	Start of the Sedov-Taylor phase
    Returns
    ----------------------
    float
        The minimum energy of a CR that is released in the ISM at time t
    """
    Eesc = Emax_t_ST0 * (t_ST0 / t) ** deltap
    return Eesc.to(u.TeV)

def calc_Ebudget(integral_data, integral_header, int_flux_max_obs):
    """
    Calculates the energy budget from the observations you want to compare to

    Parameters
    ----------------------
    integral_data : array
        The integrated gamma-ray flux data
    integral_header : str
        The header of the integrated gamma-ray flux data
    int_flux_max_obs : float
        Maximum values of your observations

    Returns
    ----------------------
    float
        The energy budget to match observations

    """

    int_flux_max_model = np.max(integral_data)
    Ebudget_model = integral_header['HEB_ERG']*u.erg

    Ebudget = (int_flux_max_obs / int_flux_max_model) * Ebudget_model

    return Ebudget


class BasePDF:
    """
    A base class for creating PDFs for both hadronic and leptonic models
    """

    def __init__(self, E_SN=1e51*u.erg):

        self.E_SN = 1e51*u.erg
   
        
    def integrate_volume(self, E, t):
        """
        Integrates the pdf over a large space (sphere of 1kpc)
        
        Parameters
        ---------------------
        E : array
            Energies to consider
        t : float
            Time to consider
        """
        
        dEs = np.diff(E)

        Rs = np.linspace(0,1000,200000)*u.pc
        dRs = np.diff(Rs)[0] * np.ones(len(Rs))
        
        self.calc_pdf(E=E, R=Rs, t=t)
        
        return np.sum(4 * np.pi * Rs * Rs * dRs * self.pdf[:, :, 0], axis=1).decompose()
