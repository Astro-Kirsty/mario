import numpy as np
from astropy import units as u
from astropy.constants import c
from spectral_cube import SpectralCube
from astropy.io import fits
from astropy.wcs import WCS
from mario.utils.model_utils import calc_dist_CR_release_point, escape_time


class GammaFlux:
    """
    A class to calculate the predicted gamma-ray flux

    """

    @u.quantity_input(nH2_cell=u.cm ** (-3))
    def __init__(self, snr, proton_cube_path, gp):

        self.snr = snr
        
        file = fits.open(proton_cube_path)
        self.header = file[0].header
        self.Jp = (file[0].data) * u.Unit(self.header['BUNIT'])
        
        self.Ep = np.logspace(self.header['CRVAL3'],
                              self.header['CRVAL3'] + (self.header['NAXIS3'] - 1) * self.header['CDELT3'],
                              self.header['NAXIS3'])* u.Unit(self.header['CUNIT3'])
        
        self.gp = gp
        self.nH2_cell = self.gp.nH2_cell()       
        
        if type(self.Ep.value) == np.ndarray:
            self.Ep = self.Ep[:, None, None]


    def calc_phi(self, Egamma):

        index = (np.abs(self.Ep - Egamma)).argmin()
        Jp = self.Jp[index:]
        Ep = self.Ep[index:]
        
        # Calculation of cross section of pp collisions
        L = np.log(Ep / u.TeV)
        s_inel = ((34.3 + 1.88 * L + 0.25 * L ** 2) * u.mbarn).to(u.cm ** 2)

        # Parameters for gamma ray spectrum
        B_g = 1.30 + 0.14 * L + 0.011 * L ** 2
        k_g = (0.801 + 0.049 * L + 0.014 * L ** 2) ** (-1)
        beta_g = (1.79 + 0.11 * L + 0.008 * L ** 2) ** (-1)
        
        x = Egamma / Ep
        x_b = x ** beta_g
    
        # Calculation of gamma ray spectra
        coeff1 = B_g * (np.log(x) / x)
        coeff2 = ((1 - x_b) / (1 + k_g * x_b * (1 - x_b))) ** 4
        coeff3 = (1 / np.log(x) - (4 * beta_g * x_b) / (1 - x_b) -
                  (4 * k_g * beta_g * x_b * (1 - 2 * x_b)) / (1 + k_g * x_b * (1 - x_b)))
        Fg = coeff1 * coeff2 * coeff3
        
        phi_g = (c * self.nH2_cell * s_inel * Jp * Fg / Ep)
        phi_g.to(1 / (u.cm ** 3 * u.TeV ** 2 * u.s))
        
        phi = np.sum(phi_g[:-1] * np.diff(Ep, axis=0), axis=0).to(1 / (u.cm ** 3 * u.TeV * u.s))


        return phi

    def flux(self, energies):

        # Calculation of the fixed depth of some box surrounding the source
        dl = self.snr.distance.to(u.cm) * np.tan(0.45 * u.deg)

        # Converting the pixel width from angular size to physical size
        pixel_width = self.snr.distance.to(u.cm) * np.tan(self.gp.theta_pixel)
        ster = (pixel_width ** 2 / (self.snr.distance.to(u.cm)) ** 2) * u.sr

        Gfluxes = []

        for Eg in energies:
            Gflux = self.calc_phi(Eg) * dl * ster / (4 * np.pi * u.sr)
            unit = Gflux.unit
            Gfluxes += [Gflux]

        self.energies = energies
        self.fluxes = Gfluxes * unit

        return Gfluxes * unit
    
    def save_cube(self, author='', filename='Gflux_cube.fits'):
        """
        Save the gamma-ray flux to a fits cube

        Parameters
        ----------------------
        author : str, optional
            Author of the fits file (default 'Kirsty Feijen')
        filename : str, optional
            The name to save your fits to (default 'Gflux_cube.fits')

        """
        step = np.diff(np.log10(self.energies.value))[0]

        # Value of the first energy in the energy array
        E0 = np.log10(self.energies[0].value)

        # Set header based on template file
        hdr = fits.PrimaryHDU(data=self.fluxes.value)
        hdr.header['BUNIT'] = f'{self.fluxes.unit}'
        
        hdr.header['CTYPE1'] = self.gp.header['CTYPE1']
        hdr.header['CRVAL1'] = self.gp.header['CRVAL1']
        hdr.header['CRPIX1'] = self.gp.header['CRPIX1']
        hdr.header['CDELT1'] = self.gp.header['CDELT1']

        hdr.header['CTYPE2'] = self.gp.header['CTYPE2']
        hdr.header['CRVAL2'] = self.gp.header['CRVAL2']
        hdr.header['CRPIX2'] = self.gp.header['CRPIX2']
        hdr.header['CDELT2'] = self.gp.header['CDELT2']

        hdr.header['CTYPE3'] = 'log ENERGY'
        hdr.header['CUNIT3'] = 'TeV'
        hdr.header['CRVAL3'] = E0
        hdr.header['CRPIX3'] = 1.
        hdr.header['CDELT3'] = step
        
        hdr.header['GDELTA'] = self.header['GDELTA']
        hdr.header['GCHI']   = self.header['GCHI']
        hdr.header['GEBUDGET'] = self.header['GEBUDGET']

        hdr.header['GAGE'] = self.header['GAGE']

        hdr.header['GMODEL'] = self.header['GMODEL']

#         hdr.header['COMMENT']  = self.header['COMMENT']

        hdr.header['AUTHOR'] = author
        hdr.writeto(filename, overwrite=True)
