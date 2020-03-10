import numpy as np
from astropy import units as u
from astropy.constants import c
from spectral_cube import SpectralCube
from astropy.io import fits
from astropy.wcs import WCS


class GammaFlux:
    """
    A class to calculate the gamma-ray flux

    """

    @u.quantity_input(dist_cell=u.cm, nH2_cell=u.cm ** (-3), theta_pixel=u.deg)
    def __init__(self, snr, ps, dist_cell, nH2_cell,
                 theta_pixel):
        """
        Parameters
        ----------------------
        snr : class
            The class for the supernova remnant counterpart
        ps : class
            The class for the proton spectra
        dist_cell : array
            The distance from the centre of the counterpart to each cell
        nH2_cell : array
            The number density of each cell
        theta_pixel : float
            The size of your cells

        Examples
        ----------------------
        >>> dist_array = gp.calc_dist()
            nH2_array = gp.calc_nH2()
            theta_pixel = gp.theta_pixel
            gf = GammaFlux(G8, ps, dist_array, nH2_array, theta_pixel)

        """
        self.snr = snr
        self.ps = ps

        self.dist_cell = dist_cell
        self.nH2_cell = nH2_cell
        self.theta_pixel = theta_pixel

        self.Egs = np.linspace(0.1, 30, 10) * u.TeV

    def calc_phi_Eg(self, Ep, Eg):
        """
        Calculates the phi value from Kelner 2006

        Parameters
        ----------------------
        Ep : array
            An array of proton energies
        Eg : array
            An array of gamma-ray energies
        """

        # Calculation of proton flux
        Jp = self.ps.proton_flux(Ep.to(u.GeV), self.dist_cell).to(u.TeV ** (-1) * u.cm ** (-3))

        if type(Ep.value) == np.ndarray:
            Ep = Ep[:, None, None]

        # Calculation of cross section of pp collisions
        L = np.log(Ep / u.TeV)
        s_inel = ((34.3 + 1.88 * L + 0.25 * L ** 2) * u.mbarn).to(u.cm ** 2)

        # Parameters for gamma ray spectrum
        B_g = 1.30 + 0.14 * L + 0.011 * L ** 2
        k_g = (0.801 + 0.049 * L + 0.014 * L ** 2) ** (-1)
        beta_g = (1.79 + 0.11 * L + 0.008 * L ** 2) ** (-1)
        x = Eg / Ep
        x_b = x ** beta_g

        # Calculation of gamma ray spectra
        coeff1 = B_g * (np.log(x) / x)
        coeff2 = ((1 - x_b) / (1 + k_g * x_b * (1 - x_b))) ** 4
        coeff3 = (1 / np.log(x) - (4 * beta_g * x_b) / (1 - x_b) -
                  (4 * k_g * beta_g * x_b * (1 - 2 * x_b)) / (1 + k_g * x_b * (1 - x_b)))
        Fg = coeff1 * coeff2 * coeff3
        phi_g = (c * self.nH2_cell * s_inel * Jp * Fg / Ep)

        return phi_g.to(1 / (u.cm ** 3 * u.TeV ** 2 * u.s))

    def calc_phi(self, Egamma):
        """
        Evaluating the intergral from calc_phi_Eg for the array of distance to cells
        and array of number densities for each cells

        """
        Eps = np.logspace(np.log10(Egamma / u.TeV).value,
                          np.log10(1000 * Egamma / u.TeV).value, 1000) * u.TeV

        Eps = Eps[1:]
        phi = np.sum([self.calc_phi_Eg(Ep, Egamma) for Ep in Eps][:-1]
                     * np.diff(Eps, axis=0)[:, None, None], axis=0)

        # Add back in units from calc_phi_Eg
        phi = phi * (1 / (u.cm ** 3 * u.TeV ** 2 * u.s))

        return phi

    def calc_Gflux(self, energies):
        """
        Calculates the gamma-ray flux

        Parameters
        ----------------------
        energies : array
            The array of gamma-ray energies you want to consider

        Examples
        ----------------------
        >>> Eg = np.logspace(-1, 2, 10)*u.TeV
        >>> fluxes = gs.calc_Gflux(Eg)

        Returns
        ----------------------
        array
            An array of gamma-ray fluxes

        """

        # Calculation of the fixed depth of some box surrounding the source
        dl = self.snr.distance.to(u.cm) * np.tan(0.45 * u.deg)

        # Converting the pixel width from angular size to physical size
        pixel_width = self.snr.distance.to(u.cm) * np.tan(self.theta_pixel)
        ster = (pixel_width ** 2 / (self.snr.distance.to(u.cm)) ** 2) * u.sr

        Gfluxes = []

        for Eg in energies:
            Gflux = self.calc_phi(Eg) * dl * ster / (4 * np.pi * u.sr)
            unit = Gflux.unit
            Gfluxes += [Gflux]

        return Gfluxes * unit

    def save_fits(self, template_path, flux, Eg, author='Kirsty Feijen', filename='Gflux_cube.fits'):
        """
        Save the gamma-ray flux to a fits cube

        Parameters
        ----------------------
        template_path : str
            The path for the template fits file
        flux :  array
            The gamma-ray flux array to save
        Eg : array
            The energies used to calculate the gamma-ray flux
        author : str, optional
            Author of the fits file (default 'Kirsty Feijen')
        filename : str, optional
            The name to save your fits to (default 'Gflux_cube.fits')

        """

        file = fits.open(template_path)
        header = file[0].header

        # Step size of energy bin
        # step = np.diff(Eg)[0].value
        step = np.diff(np.log10(Eg.value))[0]

        # Value of the first energy in the energy array
        # E0 = Eg[0].value
        E0 = np.log10(Eg[0].value)

        # Set header based on template file
        hdr = fits.PrimaryHDU(data=flux)
        hdr.header['CTYPE1'] = header['CTYPE1']
        hdr.header['CRVAL1'] = header['CRVAL1']
        hdr.header['CRPIX1'] = header['CRPIX1']
        hdr.header['CDELT1'] = header['CDELT1']

        hdr.header['CTYPE2'] = header['CTYPE2']
        hdr.header['CRVAL2'] = header['CRVAL2']
        hdr.header['CRPIX2'] = header['CRPIX2']
        hdr.header['CDELT2'] = header['CDELT2']

        hdr.header['CTYPE3'] = 'ENER'
        hdr.header['CUNIT3'] = 'J'
        hdr.header['CRVAL3'] = E0
        hdr.header['CRPIX3'] = 1.
        hdr.header['CDELT3'] = step

        hdr.header['AUTHOR'] = author
        wcs = WCS(hdr.header)

        cube = SpectralCube(data=flux, wcs=wcs)
        cube.write(filename, format='fits', overwrite=True)

        return
