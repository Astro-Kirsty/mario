import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import SkyCoord
from scipy.integrate import quad


def N0_eval(x, alpha):
    return x ** (1 - alpha)


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


class ProtonSpectrum:
    """
    A class used to determine the cosmic ray proton spectra
    of a counterpart of a gamma-ray sources

    """

    @u.quantity_input(nH=u.cm ** (-3))
    def __init__(self, gamma_src, snr, nH, diff_chi=0.01, diff_delta=0.5,
                 tsedov=100 * u.yr, Ec=150 * u.TeV, Ec_max=500 * u.TeV, delta=0.5,
                 Ebudget=1e50 * u.erg):
        """
        Parameters
        ----------------------
        gamma_src : class
            The gamma-ray source class
        snr : class
            The supernova remnant counterpart class
        nH : float
            The number density of the region surrounding the gamma-ray source
        diff_chi : float, optional
            The diffusion suppression factor (default is 0.01)
        diff_delta : float, optional
            The index of diffusion coefficient (default is 0.5)
        tsedov : float, optional
            The time of the Sedov phase (default is 100 yr)
        Ec : float, optional
            Energy of cosmic ray proton escaping the accelerator (default is 150 TeV)
        Ec_max : float, optional
            Maximum injection energy of cosmic ray protons (default is 500 TeV)
        delta : float, optional
            Describes the energy-dependent release of cosmic rays (default is 0.5)
        Ebudget : float, optional
            Energy budget of cosmic ray protons (default is 1e50 erg)

        Examples
        ----------------------
        >>> ps = ProtonSpectrum(gamma_src=hess, snr=G8, nH=160*u.cm**(-3))

        """
        self.gamma_src = gamma_src
        self.snr = snr
        self.nH = nH
        self.diff_chi = diff_chi
        self.diff_delta = diff_delta
        self.tesc = escape_time(tsedov=tsedov, Ec=Ec, Ec_max=Ec_max, delta=delta)
        self.Ebudget=Ebudget

    #@u.quantity_input(cloud_coord=u.deg, d=u.pc)
    def define_cloud(self, cloud_coord):
        """
        Calculates the distance from the cloud to the release point of cosmic ray protons

        Parameters
        ----------------------
        cloud_coord : array
            The galactic coordinates of the cloud

         Examples
        ----------------------
        >>> ProtonSpectrum.dist_cloud(g=[8.744205, -0.096288]*u.deg, gc=[8.575, -0.24]*u.deg)

        """

        snr = SkyCoord(self.snr.glon, self.snr.glat, frame='galactic')
        cloud = SkyCoord(cloud_coord[0], cloud_coord[1], frame='galactic')
        cloud_source_sep = (snr.separation(cloud)).deg * u.deg

        R = calc_dist_CR_release_point(E51=1, n=1 * u.cm ** (-3),
                                       u1=1.4 * u.kg, tesc=self.tesc)

        ang_dist = np.arctan(R / (self.snr.distance.to(u.pc))).to(u.deg)
        theta_cloud = cloud_source_sep - ang_dist

        self.dist_cloud = (self.snr.distance * np.tan(theta_cloud)).to(u.pc)

    @u.quantity_input(Ep=u.GeV, rcloud=u.pc, B_mag=u.uG)
    def flux(self, Ep, rcloud, B_mag=None):
        """
        Calculates the proton flux of your counterpart

        Parameters
        ----------------------
        Ep : array
            The energy of cosmic ray protons
        rcloud : float
            The distance from the cloud to the release point of cosmic ray protons
        B_mag : float, optional
            The magentic field of the region surrounding the counterpart

        Returns
        ----------------------
        array
            An array of values for proton-flux

        Examples
        ----------------------
        >>> Ep = np.logspace(0, 6, 50)*u.GeV
        >>> rc = 12*u.pc
        >>> ProtonSpectrum.flux(Ep, rc)

        """

        if type(Ep.value) == np.ndarray:
            Ep = Ep[:, None, None]
            if type(rcloud.value) == np.ndarray:
                rcloud = rcloud * np.ones((len(Ep), np.shape(rcloud)[0], np.shape(rcloud)[1]))

        if B_mag is None:
            B_mag = calc_Bmag(self.nH)

        # Calculation of diffusion coefficient
        D0 = 3e27 * u.cm ** 2 / u.s
        diff_coeff = self.diff_chi * D0 * (Ep / u.GeV / (B_mag / (3 * u.uG))) ** self.diff_delta

        # Calculation of cooling time of protons
        cool_time = (6e7 * (self.nH / u.cm ** -3) ** (-1) * u.yr).to(u.s)

        # Calculation of diffusion radius
        frac = ((np.exp(self.snr.age * self.diff_delta / cool_time) - 1)
                / (self.snr.age * self.diff_delta / cool_time))
        diff_radius = 2 * np.sqrt(diff_coeff * self.snr.age * frac)

        # Calculation of normalisation constant, N0
        alpha = 2
        t0 = 1 * u.yr
        R = calc_dist_CR_release_point(E51=1, n=1 * u.cm ** (-3), u1=1.4 * u.kg, tesc=t0)

        intt, err = quad(N0_eval, 1, 1e5, args=alpha)

        V = (4 / 3) * np.pi * R ** 3
        N0 = ((np.pi ** (3 / 2) * R ** 3 * (self.Ebudget.to(u.GeV)) / u.GeV) /
              (np.exp(- ((alpha - 1) * t0 / cool_time) - 1) * V * intt))

        # Calculation of proton flux
        coeff = (N0 * (Ep / u.GeV) ** (-self.gamma_src.spectral_index)) / (np.pi ** (3 / 2) * diff_radius ** 3)
        expo = (- ((self.gamma_src.spectral_index - 1) * self.snr.age / cool_time) - (rcloud / diff_radius) ** 2)
        flux = (coeff / u.GeV * np.exp(expo)).to(u.TeV ** (-1) * u.cm ** (-3))

        self.energies = Ep
        self.fluxes = flux

        return flux
