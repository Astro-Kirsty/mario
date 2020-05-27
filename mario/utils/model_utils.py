import numpy as np
from astropy import units as u
from spectral_cube import SpectralCube
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


@u.quantity_input(n=u.cm ** (-3))
def calc_Bmag(n):
    """
    Calculation of magnetic field from Crutcher et al. 2010 relation

    Parameters
    ----------------------
    n : float
        The number density of the medium
    """
    B0 = 10 * u.uG
    n0 = 300 * u.cm ** (-3)
    if n < n0:
        B_mag = B0
    else:
        B_mag = B0 * (n / n0) ** 0.65
    return B_mag


@u.quantity_input(n=u.cm ** (-3), u1=u.kg, tesc=u.s)
def calc_dist_CR_release_point(E51, n, u1, tesc):
    """
    Calculates the distance at which cosmic ray protons are released from the
    accelerator (Reynolds 2008)

    Parameters
    ----------------------
    E51 : float
        The kinetic energy in units of 10^51 erg
    n : float
        The number density in units cm^-3
    u1 : float
        Mean mass per particle
    tesc : float
        The escape time of particles

    Returns
    ----------------------
    float
        The distance at which cosmic ray protons are released
    """
    R = (0.31 * E51 ** (1 / 5) * (n / u.cm ** (-3)) ** (-1 / 5) *
         (u1 / (1.4 * u.kg)) ** (-1 / 5) * (tesc / u.yr) ** (2 / 5)) * u.pc
    return R


@u.quantity_input(tsedov=u.s, Ec=u.TeV, Ec_max=u.TeV)
def escape_time(tsedov, Ec, Ec_max, delta):
    """
    Calculates the escape time of cosmic ray protons (eg Gabici et al. 2009)

    Parameters
    ----------------------
    tsedov : float
        The time of the Sedov phase
    Ec : float
        Energy of cosmic ray proton escaping the accelerator
    Ec_max : float
        Maximum injection energy of cosmic ray protons
    delta : float
        Describes the energy-dependent release of CRs

    Returns
    ----------------------
    float
        The escape time of cosmic ray protons

    """
    tesc = tsedov * (Ec / Ec_max) ** (-1 / delta)
    return tesc
