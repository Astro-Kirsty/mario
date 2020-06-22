import numpy as np
from astropy import units as u
from astropy.constants import c
from mario.utils.model_utils import calc_Bmag, calc_dist_CR_release_point, escape_time

class ElectronSpectrum:
    """
    A class used to determine the cosmic ray electron spectra
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
        >>> es = ElectronSpectrum(gamma_src=hess, snr=G8, nH=160*u.cm**(-3))

        """
        self.gamma_src = gamma_src
        self.snr = snr
        self.nH = nH
        self.diff_chi = diff_chi
        self.diff_delta = diff_delta

        self.tsedov = tsedov
        self.Ec = Ec
        self.Ec_max = Ec_max
        self.deltap = deltap
        self.n0 = n0
        
        self.E51 = 1
        self.u1 = 1.4 * u.kg

        self.Ebudget = Ebudget

        self.model = 'ElectronSpectrum'

    def define_cloud(self, cloud_coord, Ec=None):
        """
        Calculates the distance from the cloud to the release point of cosmic ray protons

        Parameters
        ----------------------
        cloud_coord : array
            The galactic coordinates of the cloud

         Examples
        ----------------------
        >>> ElectronSpectrum.dist_cloud(g=[8.744205, -0.096288]*u.deg, gc=[8.575, -0.24]*u.deg)

        """
        
        if Ec is None:
            Ec = self.Ec

        snr = SkyCoord(self.snr.glon, self.snr.glat, frame='galactic')
        cloud = SkyCoord(cloud_coord[0], cloud_coord[1], frame='galactic')
        cloud_source_sep = (snr.separation(cloud)).deg * u.deg

        tesc = escape_time(tsedov=self.tsedov, Ec=Ec, Ec_max=self.Ec_max, delta=self.deltap)
        
        Rc = calc_dist_CR_release_point(E51=self.E51, n=self.n0, u1=self.u1, tesc=tesc)

        Rc_ang_dist = np.arctan(Rc / (self.snr.distance.to(u.pc))).to(u.deg)
        theta_cloud = cloud_source_sep - Rc_ang_dist

        self.dist_cloud = (self.snr.distance * np.tan(theta_cloud)).to(u.pc)

    @u.quantity_input(Ee=u.GeV, rcloud=u.pc, B_mag=u.uG)
    def flux(self, Ee, rcloud, B_mag=None):
        """
        Calculates the electron flux of your counterpart

        Parameters
        ----------------------
        Ee : array
            The energy of electrons
        rcloud : float
            The distance from the cloud to the release point of cosmic ray protons
        B_mag : float, optional
            The magentic field of the region surrounding the counterpart

        Returns
        ----------------------
        array
            An array of values for electron-flux

        Examples
        ----------------------
        >>> Ee = np.logspace(-3, 6, 50)*u.GeV
        >>> rc = 10*u.pc
        >>> ElectronSpectrum.flux(Ee, rc)

        """

        if type(Ee.value) == np.ndarray:
            Ee = Ee[:, None, None]
            if type(rcloud.value) == np.ndarray:
                rcloud = rcloud * np.ones((len(Ee), np.shape(rcloud)[0], np.shape(rcloud)[1]))

        if B_mag is None:
            B_mag = calc_Bmag(self.nH)
            

        # Calculation of diffusion coefficient
        me = (511 * u.keV).to(u.GeV) /c**2
        D0 = 3e27 * u.cm ** 2 / u.s
        diff_coeff = self.diff_chi * D0 * ( (Ee / (me*c**2)) / (B_mag / (3 * u.uG))) ** self.diff_delta

        # Synchrotron parameter
        bs = 1.292e-15 / u.s * (B_mag.to(u.mG) / u.mG)**2 

        # Calculation of diffusion radius
        frac = 4*diff_coeff / (bs * (Ee / (me*c**2)) * (1 - self.diff_delta))
        brac = 1 - (1 - (Ee / (me*c**2)) * bs * self.snr.age)**(1 - self.diff_delta)
        diff_radius = np.sqrt(frac * brac)
        
        # Calculation of normalisation constant, N0
        t0 = 1 * u.yr
        R = calc_dist_CR_release_point(E51=self.E51, n=self.n0, u1=self.u1, tesc=t0)

        V = (4 / 3) * np.pi * R ** 3
        intg = (1e5**2 - 1**2) /2
        N0 = (self.Ebudget.to(u.GeV)/u.GeV * np.pi ** (3 / 2) * R ** 3) / (V * intg)
        
        # Calculation of electron flux
        coeff = N0 * (1 - (Ee / (me*c**2)) * bs * self.snr.age)**(self.gamma_src.spectral_index-2) / (np.pi ** (3 / 2) * diff_radius ** 3)
        expo = - rcloud**2 / diff_radius**2
        flux = coeff * np.exp(expo)

        self.energies = Ee
        self.fluxes = flux

        return flux
