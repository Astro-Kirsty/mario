import random
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from scipy.optimize import curve_fit
from spectral_cube import SpectralCube
from gammapy.spectrum import SpectrumDataset
from gammapy.modeling.models import (
    PowerLawSpectralModel, SkyModel, SmoothBrokenPowerLawSpectralModel,
    LogParabolaSpectralModel)
from gammapy.irf import load_cta_irfs


class GammaCounts:
    """
    A class for simulating gamma-ray counts
    """

    @u.quantity_input(livetime=u.hr, offset=u.deg, energy=u.TeV)
    def __init__(self, cube_path, cta_irf_path, hess_irf_path, livetime, offset, energy):
        """
        Parameters
        ----------------------
        cube_path : str
            Path to the gamma-ray flux cube in the format of fits
        cta_irf_path : str
            Path to the instrument response function for the Cherenkov
            Telescope Array in the format of fits
        hess_irf_path : str
            Path to the instrument response function for the High Energy Sterescopic System
        livetime : float
            The livetime of the telescope
        offset : float
            The offset for the telescope observations
        energy : array
            The energy array for the Cherenkov Telescope Array observations

        Examples
        ----------------------
        >>> irf_hess = '/home/kirsty/Documents/PhD/CTA/Data_challenge/hess_dl3_dr1'
        >>> irf_cta = '$CTADATA/1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
        >>> cube_path = 'Fits/Gflux_all_cube.fits'
        >>> energy = np.logspace(-1, 2, 31)*u.TeV
        >>> gc = GammaCounts(cube_path, irf_cta, irf_hess, 15.7*u.hr, 0.5*u.deg, energy)
        >>> gc.calc_counts_CTA()

        """
        self.cube = SpectralCube.read(cube_path)
        self.livetime = livetime
        self.offset = offset
        self.energy = energy

        self.hess_irf = DataStore.from_dir(hess_irf_path)

        self.cta_irf = load_cta_irfs(cta_irf_path)

    def func(self, x, a, b):
        """ Setting up the powerlaw function to fit """
        return a + b * x

    def parameters(self, y_cell, x_cell):
        """
        Calculates the amplitude and index from a gamma-ray flux array

        Parameters
        ----------------------
        y_cell : int
            y cell values to loop over
        x_cell : int
            x cell values to loop over
        """
        # Must have energy array same as z-axis of the gamma-ray flux
        nsize = self.cube.spectral_axis.size
        Emin = self.cube.spectral_axis[0].value
        Emax = self.cube.spectral_axis[-1].value
        energies = np.logspace(Emin, Emax, nsize)

        fluxes = self.cube[:, y_cell, x_cell].value
        logx = np.log10(energies)

        for m in range(len(fluxes)):
            if fluxes[m] == 0:
                amplitude = 0
                index = random.uniform(-0.1, -4)
            else:
                logy = np.log10(fluxes)
                popt, pcov = curve_fit(self.func, logx, logy, [logy[1], 0.3])
                amplitude = 10 ** popt[0]
                index = popt[1]

        return amplitude, index

    def calc_counts_CTA(self):
        """
        Calculates the simulated excess counts map for the Cherenkov Telescope Array

        Returns
        ----------------------
        array
            The excess counts array
        """

        E, xval, yval = self.cube.shape

        aeff = self.cta_irf['aeff'].to_effective_area_table(offset=self.offset, energy=self.energy)
        edisp = self.cta_irf['edisp'].to_energy_dispersion(offset=self.offset,
                                                           e_true=self.energy, e_reco=self.energy)

        params = np.empty((xval, yval, 2))
        counts_CTA = np.empty((xval, yval))
        for ix in range(xval):
            for jy in range(yval):
                params[ix, jy, :] = self.parameters(ix, jy)
                model_sim = PowerLawSpectralModel(
                    index=-params[ix, jy, 1],
                    amplitude=params[ix, jy, 0] * u.Unit('cm-2 s-1 TeV-1'),
                    reference=1 * u.TeV)
                model_ref = SkyModel(spectral_model=model_sim)
                dataset = SpectrumDataset(aeff=aeff, edisp=edisp, models=model_ref, livetime=self.livetime)
                dataset.fake(random_state=42)
                counts_CTA[ix, jy] = dataset.counts.data.sum()

        return counts_CTA

    def calc_counts_HESS(self, obs_ID=20900):
        """
        Calculates the simulated excess counts map for the High Energy Stereoscopic System

        Parameters
        ----------------------
        obs_ID : int, optional
            The observation ID for HESS observation (default is 20900)

        Returns
        ----------------------
        array
            The excess counts array
        """

        E, xval, yval = self.cube.shape

        HESS_obs = self.hess_irf.obs(obs_ID)
        aeff = HESS_obs.aeff.to_effective_area_table(offset=self.offset, energy=self.energy)
        edisp = HESS_obs.edisp.to_energy_dispersion(offset=self.offset,
                                                    e_true=self.energy, e_reco=self.energy)

        params = np.empty((xval, yval, 2))
        counts_HESS = np.empty((xval, yval))
        for ix in range(xval):
            for jy in range(yval):
                params[ix, jy, :] = self.parameters(ix, jy)
                model_sim = PowerLawSpectralModel(
                    index=-params[ix, jy, 1],
                    amplitude=params[ix, jy, 0] * u.Unit('cm-2 s-1 TeV-1'),
                    reference=1 * u.TeV)
                model_ref = SkyModel(spectral_model=model_sim)
                dataset = SpectrumDataset(aeff=aeff, edisp=edisp, models=model_ref, livetime=self.livetime)
                dataset.fake(random_state=42)
                counts_HESS[ix, jy] = dataset.counts.data.sum()

        return counts_HESS

    def save_fits(self, template_path, counts, author='Kirsty Feijen', filename='counts.fits'):
        """
        Save the counts map to a fits file

        Parameters
        ----------------------
        template_path : str
            The path for the template fits file
        counts :  array
            The counts array to save
        author : str, optional
            Author of the fits file (default 'Kirsty Feijen')
        filename : str, optional
            The name to save your fits to (default 'counts.fits')

        """
        file = fits.open(template_path)
        header = file[0].header

        # Set header based on template file
        hdr = fits.PrimaryHDU(data=counts)
        hdr.header['CTYPE1'] = header['CTYPE1']
        hdr.header['CRVAL1'] = header['CRVAL1']
        hdr.header['CRPIX1'] = header['CRPIX1']
        hdr.header['CDELT1'] = header['CDELT1']

        hdr.header['CTYPE2'] = header['CTYPE2']
        hdr.header['CRVAL2'] = header['CRVAL2']
        hdr.header['CRPIX2'] = header['CRPIX2']
        hdr.header['CDELT2'] = header['CDELT2']

        hdr.header['AUTHOR'] = author
        # wcsr = WCS(hdr.header)

        hdr.writeto(filename, overwrite=True)

        return
