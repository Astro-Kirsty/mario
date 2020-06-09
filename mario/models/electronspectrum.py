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
                 tsedov=100 * u.yr, Ec=150 * u.TeV, Ec_max=500 * u.TeV, delta=0.5,
                 Ebudget=1e50 * u.erg):
        """


        """
        self.gamma_src = gamma_src
        self.snr = snr
        self.nH = nH
        self.diff_chi = diff_chi
        self.diff_delta = diff_delta
        self.tesc = escape_time(tsedov=tsedov, Ec=Ec, Ec_max=Ec_max, delta=delta)
        self.Ebudget=Ebudget

    def define_cloud(self, cloud_coord):
        """

        """

        snr = SkyCoord(self.snr.glon, self.snr.glat, frame='galactic')
        cloud = SkyCoord(cloud_coord[0], cloud_coord[1], frame='galactic')
        cloud_source_sep = (snr.separation(cloud)).deg * u.deg

        R = calc_dist_CR_release_point(E51=1, n=1 * u.cm ** (-3),
                                       u1=1.4 * u.kg, tesc=self.tesc)

        ang_dist = np.arctan(R / (self.snr.distance.to(u.pc) / u.pc)).to(u.deg)
        theta_cloud = cloud_source_sep - ang_dist

        self.dist_cloud = (self.snr.distance * np.tan(theta_cloud)).to(u.pc)

    @u.quantity_input(Ee=u.GeV, rcloud=u.pc, B_mag=u.uG)
    def flux(self, Ee, rcloud, B_mag=None):
        """


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
        R = calc_dist_CR_release_point(E51=1, n=1 * u.cm ** (-3), u1=1.4 * u.kg, tesc=t0)
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
