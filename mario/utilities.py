import numpy as np
import astropy.units as u

@u.quantity_input(n=u.cm**(-3))
def calc_Bmax(n):
    """
    Calculation of the maximum magnetic field in a cloud 
    from Crutcher et al. 2010

    Parameters
    ----------------------
    n : float
        The number density of the cloud
    """
    B0 = 10*u.uG
    n0 = 300*u.cm**(-3)
    
    if n < n0:
        B_max = B0
    else:
        B_max = B0 * (n / n0) ** 0.65
        
    return B_max

@u.quantity_input(E_SN=u.erg, n0=u.cm**(-3), t=u.yr)
def calc_shock_radius(t, E_SN=1e51*u.erg, n0=1*u.cm**-3):
    """
    Calculates the radius of the shock of a supernova remnant in the Sedov-Taylor phase.
    It can be used as the release point of cosmic-ray protons from the accelerator, 
    if t corresponds to the escape time.
    (Reynolds 2008 or Truelove & McKee 1999)

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
    R = 0.31 * u.pc * ((E_SN / (1e51*u.erg)) / (n0 / u.cm**-3)).decompose() ** (1/5) * (t / u.yr) ** (2 / 5)
    return R.to(u.pc)

@u.quantity_input(Emax_100yr=u.TeV, Emax_t=u.TeV, t=u.yr)
def calc_energy_evolution(t=50*u.kyr, Emax_t=1*u.GeV, Emax_100yr=1*u.PeV):
    """
    Calculates the energy evolution.
    This parameter is used to calculate the time after supernova explosion
    at which CRs with energy E are released in the ISM
    (eg Gabici et al. 2009)

    Parameters
    ----------------------
    t : float
        Time after supernova explosion at which particles 
        of maximum energy Emax_t are accelerated
    Emax_t : float
        Maximum energy of particles at time t after supernova explosion
    Emax_100yr : float
        Maximum energy of particles at 100yr after supernova explosion
        (corresponds to start of Sedov-Taylor phase)

    Returns
    ----------------------
    float
        The energy evolution parameter 
        (often referred to as 'delta')

    """

    return np.log(Emax_t / Emax_100yr) / np.log(100*u.yr / t)


@u.quantity_input(E=u.TeV, Emax_100yr=u.TeV)
def calc_escape_time(E, delta, Emax_100yr=1*u.PeV):
    """
    Calculates the time after supernova explosion at which CRs
    with energy E are released in the ISM
    (eg Gabici et al. 2009)

    Parameters
    ----------------------
    E: float
        The energy of CRs that are released in the ISM
    delta : float
        Energy evolution parameter
    Emax_100yr : float
        Maximum energy of particles at 100yr after supernova explosion
        (corresponds to start of Sedov-Taylor phase)

    Returns
    ----------------------
    float
        The time after supernova explosion at which CRs are released

    """
    tesc = 100*u.yr * (E / Emax_100yr) ** (- 1 / delta)
    return tesc.to(u.kyr)

@u.quantity_input(t=u.yr, Emax_100yr=u.TeV)
def calc_escape_energy(t, delta, Emax_100yr=1*u.PeV):
    """
    Calculates the minimum energy at which CRs
    are released in the ISM at time t
    (eg Gabici et al. 2009)

    Parameters
    ----------------------
    t: float
        The time at which CRs are released in the ISM
    delta : float
        Energy evolution parameter
    Emax_100yr : float
        Maximum energy of particles at 100yr after supernova explosion
        (corresponds to start of Sedov-Taylor phase)

    Returns
    ----------------------
    float
        The minimum energy of a CR that is released in the ISM at time t

    """
    Eesc = Emax_100yr * (100*u.yr / t) ** delta
    return Eesc.to(u.TeV)


class BasePDF:
    """
    A base class for creating PDFs for both hadronic and leptonic models
    """

    def __init__(
        self, 
        E_SN=1e51*u.erg,
    ):

        self.E_SN = 1e51*u.erg
   
        
    def integrate_volume(self, E, t):
        """
        Integrates the pdf over a large space (sphere of 20kpc)
        
        Parameters
        ---------------------
        E : array
            Energies to consider
        t : float
            Time to consider
        """
        
        dEs = np.diff(E)

        Rs = np.linspace(0,20000,200000)*u.pc
        dRs = np.diff(Rs)[0] * np.ones(len(Rs))
        
        self.calc_pdf(E=E, R=Rs, t=t)
        
        return np.sum(4 * np.pi * Rs * Rs * dRs * self.pdf[:, :, 0], axis=1).decompose()