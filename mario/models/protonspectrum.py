import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import SkyCoord
from scipy.integrate import quad
from mario.utils.model_utils import calc_Bmag, calc_dist_CR_release_point, escape_time

def N0_eval(x, alpha):
    return x ** (1 - alpha)


class ProtonSpectrum:
    """
    A class used to determine the cosmic ray proton spectra
    of a counterpart of a gamma-ray sources

    """

    @u.quantity_input(nH=u.cm ** (-3))
    def __init__(self, gamma_src, snr, nH, diff_chi=0.01, diff_delta=0.5,
                 tsedov=100 * u.yr, Ec=150 * u.TeV, Ec_max=500 * u.TeV, deltap=0.5, n0= 1 * u.cm ** (-3),
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
        deltap : float, optional
            Describes the energy-dependent release of cosmic rays (default is 0.5)
        n0 : float, optional
            Number density at time of cosmic ray release (default is 1 cm^-3)
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

        self.time = self.snr.age

        self.tsedov = tsedov
        self.Ec = Ec
        self.Ec_max = Ec_max
        self.deltap = deltap
        self.n0 = n0
        
        self.E51 = 1
        self.u1 = 1.4 * u.kg

        self.Ebudget = Ebudget

    #@u.quantity_input(cloud_coord=u.deg, d=u.pc)
    def define_cloud(self, cloud_coord, Ec=None):
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
        
        if Ec is None:
            Ec = self.Ec

        snr = SkyCoord(self.snr.glon, self.snr.glat, frame='galactic')
        cloud = SkyCoord(cloud_coord[0], cloud_coord[1], frame='galactic')
        cloud_source_sep = (snr.separation(cloud)).deg * u.deg

        tesc = escape_time(tsedov=self.tsedov, Ec=Ec, Ec_max=self.Ec_max, delta=self.deltap)
        
        R = calc_dist_CR_release_point(E51=self.E51, n=self.n0, u1=self.u1, tesc=tesc)

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
        frac = ((np.exp(self.time * self.diff_delta / cool_time) - 1)
                / (self.time * self.diff_delta / cool_time))
        diff_radius = 2 * np.sqrt(diff_coeff * self.time * frac)
               
        # Calculation of normalisation constant, N0
        alpha = 2
        t0 = 1 * u.yr
        R = calc_dist_CR_release_point(E51=self.E51, n=self.n0, u1=self.u1, tesc=t0)
        
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


class TimeDependentProtonSpectrum(ProtonSpectrum):

    def __init__(self, gamma_src, snr, nH, Ep, diff_chi=0.01, diff_delta=0.5,
                 tsedov=100 * u.yr, Ec=150 * u.TeV, Ec_max=500 * u.TeV, deltap=1.0,
                 Ebudget=1e50 * u.erg, n0=0.03 * u.cm ** (-3)):
        super(TimeDependentProtonSpectrum, self).__init__(gamma_src, snr, nH, diff_chi, diff_delta,
                                                          tsedov, Ec, Ec_max, deltap, n0, Ebudget)

        self.Ep = Ep

        self.tesc = escape_time(tsedov=self.tsedov, Ec=self.Ep.to(u.TeV), 
                                Ec_max=self.Ec_max, delta=self.deltap)
        self.time = (self.snr.age - self.tesc)
        self.time[self.tesc > self.snr.age] = 0.05*u.kyr
        
        if type(self.time.value) == np.ndarray:
            self.time = self.time[:, None, None]
            
        self.Resc = calc_dist_CR_release_point(E51=self.E51, n=self.n0, u1=self.u1, tesc=self.tesc)


    def escape_energy(self):
    
        # Energy of gamma-rays at the radius of the SNR
        r_snr = self.snr.distance.to(u.pc) * np.tan(self.snr.radius)
        tesc_snr = ( r_snr / (0.31*u.pc * self.E51**(1/5) * (self.n0/u.cm**(-3))**(-1/5) * 
                              (self.u1/ (1.4*u.kg))**(-1/5)))**(5/2) * u.yr
        Eg_esc = (self.Ec_max * (tesc_snr / self.tsedov)**(-self.deltap) ) * 0.17

        self.Eg_esc = Eg_esc
        self.tesc_snr = tesc_snr

        return 



