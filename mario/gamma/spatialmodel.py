import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit
from gammapy.spectrum import SpectrumDataset
from gammapy.modeling.models import (PowerLawSpectralModel, LogParabolaSpectralModel,
                                     ExpCutoffPowerLawSpectralModel, SkyModel)
from photutils import SkyCircularAperture
from astropy.wcs import WCS
import matplotlib.pyplot as plt


def oversample(data, header, grid_size=0.005 * u.deg, radius=0.06 * u.deg, diff_vals=True):
    """
    Oversample a map

    Parameters
    ----------------------
    data : array
        The data for oversampling
    header : str
        The header of the fits file
    grid_size : float, optional
        Size of grid pixels (default is 0.005deg)
    aperture_size : float, optional
        Size of aperture you want to smooth (default is 0.06deg)
	diff_vals : bool, optional
		If the fits file header contains the diffusion parameters (default is True)

    """
    wcs = WCS(header)

    glon_last, glat_first = wcs.all_pix2world(0, 0, 0)
    glon_first, glat_last = wcs.all_pix2world(data.shape[1], data.shape[0], 0)
    full_glon = np.arange(glon_last, glon_first, -grid_size.to(u.deg) / u.deg) * u.deg
    full_glat = np.arange(glat_first, glat_last, grid_size.to(u.deg) / u.deg) * u.deg

    l, b = np.meshgrid(full_glon, full_glat)
    position = SkyCoord(l=l, b=b, frame='galactic').flatten()
    aperture = SkyCircularAperture(position, radius).to_pixel(wcs)
    mask = aperture.to_mask(method='center')

    data_oversample = []
    for m in range(len(mask)):
        data_oversample += [np.sum(mask[m].multiply(data))]
    data_oversample = np.array(data_oversample).reshape((len(full_glat), len(full_glon)))

    hdu = fits.PrimaryHDU(data=data_oversample)
    hdu.header['NAXIS1'] = data_oversample.shape[1]
    hdu.header['NAXIS2'] = data_oversample.shape[0]
    hdu.header['CTYPE1'] = header['CTYPE1']
    hdu.header['CRVAL1'] = header['CRVAL1']
    hdu.header['CRPIX1'] = header['CRPIX1']

    hdu.header['CDELT1'] = -grid_size.value
    hdu.header['CTYPE2'] = header['CTYPE2']
    hdu.header['CRVAL2'] = header['CRVAL2']
    hdu.header['CRPIX2'] = data_oversample.shape[0] / 2
    hdu.header['CDELT2'] = grid_size.value

    if diff_vals:
        hdu.header['GDELTA'] = header['GDELTA']
        hdu.header['GCHI'] = header['GCHI']
        hdu.header['GEBUDGET'] = header['GEBUDGET']

    return hdu.header, data_oversample


def PL_log(E, amplitude, index):
    """
    Powerlaw fit for the gamma-ray flux array
    log(y) = log(a) - b*log(x)

    Parameters
    ----------------------
    E : array
        The array of energies to fit your data to
    amplitude : float
        The amplitude of the powerlaw fit
    index : float
        The index of the powerlaw fit

    """
    return amplitude - index * E


def LP_log(E, amplitude, alpha, beta):
    """
    Log parabola fit for the gamma-ray flux array
    log(y) = log(a) - alpha*log(x) - beta*log(x)*log(x)

    Parameters
    ----------------------
    E : array
        The array of energies to fit your data to
    amplitude : float
        The amplitude of the log parabola fit
    alpha : float
        The index of the log parabola fit
    beta : float
        The curvature of the log parabola fit

    """
    return amplitude - alpha * E - beta * E * E


class SpectralModel:
    def __init__(self, energies, fluxes):
        self.energies = energies
        self.fluxes = fluxes

        self.counts = None

    def fit(self, fit='PL'):
        """
        Fit the spectral model

        Parameters
        ----------
        fit : str
            Choose which function to fit (default is 'PL')
        """
        # NOTE: if the flux has units must use .value
        log_fluxes = np.log10(self.fluxes)
        log_energies = np.log10(self.energies)

        if fit == 'PL':
            popt, pcov = curve_fit(PL_log, log_energies, log_fluxes, p0=[-16, 3])
            fit_data = PL_log(log_energies, popt[0], popt[1])
        if fit == 'LP':
            popt, pcov = curve_fit(LP_log, log_energies, log_fluxes, p0=[log_fluxes[0], 3, 0])
            fit_data = LP_log(log_energies, popt[0], popt[1], popt[2])

        self.fit_data = fit_data

        return popt

    def plot(self, fig=None, ax=None, **kwargs):
        """
        Plot the spectral model

        Parameters
        ----------
        fig : str
            Set the figure for the plot
        ax : str
            Set the axis for the figure
        kwargs : keywords
            Initialisation keywords
        """
        if fig is None:
            fig = plt.figure(figsize=(6, 4))

        if ax is None:
            ax = fig.add_subplot(111)

        ax.plot(self.energies, self.fluxes, 'ro', **kwargs)

        ax.set_xlabel('Energy (TeV)')
        ax.set_ylabel('$\gamma$-ray flux (TeV$^{-1}$cm$^{-2}$s$^{-1}$)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(linestyle='--')
        ax.legend()

        return fig, ax

    def plot_errorbar(self, yerr=None, xerr=None, fig=None, ax=None, **kwargs):
        """
        Plot the spectral model

        Parameters
        ----------
        fig : str
            Set the figure for the plot
        ax : str
            Set the axis for the figure
        kwargs : keywords
            Initialisation keywords
        """
        if fig is None:
            fig = plt.figure(figsize=(6, 4))

        if ax is None:
            ax = fig.add_subplot(111)

        ax.errorbar(self.energies, self.fluxes, yerr, xerr, **kwargs)

        ax.set_xlabel('Energy (TeV)')
        ax.set_ylabel('$\gamma$-ray flux (TeV$^{-1}$cm$^{-2}$s$^{-1}$)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(linestyle='--')
        ax.legend()

        return fig, ax


    def fake(self, aeff, edisp, livetime, fit_params, fit='PL'):
        """
        Calculates the simulated excess counts map

        Parameters
        ----------------------
        aeff : str
            The effective area from the instrument response function
        edisp : str
            The energy dispersion from the instrument response function
        livetime : float
            The livetime of telescope observations
        fit_params : array
            The fit parameters
        fit : str
            The fit for the data (default is 'PL')

        """
        if fit == 'PL':
            if len(fit_params) != 2:
                print('This fit requires 2 fit parameters: index and amplitude')

            spectral_model = PowerLawSpectralModel(
                index=fit_params[0],
                amplitude=fit_params[1] * u.Unit('cm-2 s-1 TeV-1'),
                reference=1 * u.TeV)
        if fit == 'LP':
            if len(fit_params) != 2:
                print('This fit requires 3 fit parameters: alpha, beta and amplitude')

            spectral_model = LogParabolaSpectralModel(
                alpha=fit_params[0],
                beta=fit_params[1],
                amplitude=fit_params[2] * u.Unit('cm-2 s-1 TeV-1'),
                reference=1 * u.TeV)

        sky_model = SkyModel(spectral_model=spectral_model)
        dataset = SpectrumDataset(aeff=aeff, edisp=edisp, models=sky_model, livetime=livetime)
        counts = dataset.npred().data

        self.counts = counts


class SpatialModel:

    def __init__(self, data, header):
        self.data = data
        self.header = header

    def save_fits(self, data, header=None, filename='file.fits'):
        """
        Save the data to a fits file

        Parameters
        ----------------------
        data : array
            The data array of interest
        header : str, optional
            The header of the fits file (default is 'self.header')
        filename : str, optional
            The name to save your fits to (default is 'file.fits')

        """
        hdr = fits.PrimaryHDU(data=data)

        if header is None:
            hdr.header = self.header

        hdr.writeto(filename, overwrite=True)

    def plot(self, data, fig=None, ax=None, colorbar_label='', diff_label=False, **kwargs):

        wcs = WCS(self.header, naxis=2)

        if fig is None:
            fig = plt.figure(figsize=(15, 4))

        if ax is None:
            ax = fig.add_subplot(projection=wcs)

        lon = ax.coords['glon']
        lat = ax.coords['glat']
        lon.set_axislabel('Galactic Longitude')
        lat.set_axislabel('Galactic Latitude')

        lat.set_major_formatter('d.dd')
        lon.set_major_formatter('d.dd')
        lon.set_ticklabel_position('b')

        lat.set_ticklabel(rotation=90)
        lat.set_ticks_position('l')
        lat.set_ticklabel_position('l')
        lat.set_axislabel_position('l')

        lon.set_ticks(spacing=0.4 * u.deg, color='black')
        lat.set_ticks(spacing=0.4 * u.deg, color='black')
        lon.display_minor_ticks(True)
        lon.set_minor_frequency(4)
        lat.display_minor_ticks(True)
        lat.set_minor_frequency(4)

        img = ax.imshow(data, **kwargs)

        cb = fig.colorbar(img, pad=0.005)
        cb.set_label(colorbar_label, rotation=270, labelpad=15)

        ylim, xlim = data.shape
        ax.set_xlim([0, xlim])
        ax.set_ylim([0, ylim])

        if diff_label:
                    chi = self.header['GCHI']
                    delta = self.header['GDELTA']
                    ebudget = self.header['GEBUDGET']

                    ax.text(0.03, 0.9, 
                            f'$\chi$={chi}, $\delta$={delta}, '+'E$_{budget}$='+f'{ebudget}',
                            horizontalalignment='left',
                            transform=ax.transAxes,
                            fontsize=15, color='black')

        return fig, ax


class SpatialModel3D(SpatialModel):
    """
    A class for simulating gamma-ray counts
    """

    def __init__(self, data, header):
        super(SpatialModel3D, self).__init__(data, header)
        """
        Parameters
        ----------------------
        header : str
            The header of the gamma-ray flux data
        data : array
            The gamma-ray flux data cube

        Examples
        ----------------------
        >>> 

        """
        self.energies = np.logspace(self.header['CRVAL3'],
                                    self.header['CRVAL3'] + (self.header['NAXIS3'] - 1) * self.header['CDELT3'],
                                    self.header['NAXIS3'])

        self.amplitude_PL = None
        self.index_PL = None
        self.rsd_dof_PL = None

        self.amplitude_LP = None
        self.alpha_LP = None
        self.beta_LP = None
        self.rsd_dof_LP = None

        self.best_fit = None

    def fit(self, fit='best'):
        """ Choosing the fit for the data """
        if fit == 'best':
            self._fitPL()
            self._fitLP()
            self.best_fit = np.argmin([self.rsd_dof_LP, self.rsd_dof_PL], axis=0)
        if fit == 'PL':
            self._fitPL()
        if fit == 'LP':
            self._fitLP()

    def _fitPL(self):
        """ Fit the data with a simple powerlaw """
        Esize, xsize, ysize = self.data.shape
        amplitude_PL = np.empty((xsize, ysize))
        index_PL = np.empty((xsize, ysize))
        rsd_dof_PL = np.empty((xsize, ysize))

        log_energies = np.log10(self.energies)
        data = self.data

        for ix in range(xsize):
            for iy in range(ysize):
                fluxes = data[:, ix, iy]
                log_fluxes = np.log10(fluxes)
                if np.isfinite(log_fluxes).any():
                    popt, pcov = curve_fit(PL_log,
                                           log_energies, log_fluxes,
                                           p0=[log_fluxes[0], 3])
                    amplitude_PL[ix, iy] = 10 ** popt[0]
                    index_PL[ix, iy] = popt[1]
                    rsd_dof_PL[ix, iy] = self._rsd_dof(fluxes, 10 ** PL_log(log_energies, *popt), len(popt))
                else:
                    amplitude_PL[ix, iy] = 0
                    index_PL[ix, iy] = 100
                    rsd_dof_PL[ix, iy] = 1000

        self.index_PL = index_PL
        self.amplitude_PL = amplitude_PL
        self.rsd_dof_PL = rsd_dof_PL

    def _fitLP(self):
        """ Fit the data with a log parabola """
        Esize, xsize, ysize = self.data.shape
        amplitude_LP = np.empty((xsize, ysize))
        alpha_LP = np.empty((xsize, ysize))
        beta_LP = np.empty((xsize, ysize))
        rsd_dof_LP = np.empty((xsize, ysize))

        log_energies = np.log10(self.energies)
        data = self.data

        for ix in range(xsize):
            for iy in range(ysize):
                fluxes = data[:, ix, iy]
                log_fluxes = np.log10(fluxes)
                if np.isfinite(log_fluxes).any():
                    popt, pcov = curve_fit(LP_log,
                                           log_energies, log_fluxes,
                                           p0=[log_fluxes[0], 3, 0])
                    amplitude_LP[ix, iy] = 10 ** popt[0]
                    alpha_LP[ix, iy] = popt[1]
                    beta_LP[ix, iy] = popt[2]
                    rsd_dof_LP[ix, iy] = self._rsd_dof(fluxes, 10 ** LP_log(log_energies, *popt), len(popt))
                else:
                    amplitude_LP[ix, iy] = 0
                    alpha_LP[ix, iy] = 100
                    beta_LP[ix, iy] = 100
                    rsd_dof_LP[ix, iy] = 1000

        self.amplitude_LP = amplitude_LP
        self.alpha_LP = alpha_LP
        self.beta_LP = beta_LP
        self.rsd_dof_LP = rsd_dof_LP

    def _rsd_dof(self, obs, fit, dof):
        """
        Calculates the relative square distance divided by the degrees of freedom

        Parameters
        ----------------------
        obs : array
            The observations
        fit : array
            The fit to the observations
        dof : int
            The number of degrees of freedom in the fit

        """
        rsd = np.sum((obs - fit) ** 2 / obs)
        return rsd / dof

    def fake(self, aeff, edisp, livetime, fit='best'):
        """
        Calculates the simulated excess counts map

        Parameters
        ----------------------
        aeff : str
            The effective area from the instrument response function
        edisp : str
            The energy dispersion from the instrument response function
        livetime : float
            The livetime of telescope observations
        fit : str
            The fit for the data (default is 'best')

        """

        Esize, xsize, ysize = self.data.shape
        counts = np.empty((xsize, ysize))

        for ix in range(xsize):
            for iy in range(ysize):
                if ((fit == 'best') & (self.best_fit[ix, iy] == 0)) | (fit == 'PL'):
                    spectral_model = PowerLawSpectralModel(
                        index=self.index_PL[ix, iy],
                        amplitude=self.amplitude_PL[ix, iy] * u.Unit('cm-2 s-1 TeV-1'),
                        reference=1 * u.TeV)
                elif ((fit == 'best') & (self.best_fit[ix, iy] == 1)) | (fit == 'LP'):
                    spectral_model = LogParabolaSpectralModel(
                        alpha=self.alpha_LP[ix, iy],
                        beta=self.beta_LP[ix, iy],
                        amplitude=self.amplitude_LP[ix, iy] * u.Unit('cm-2 s-1 TeV-1'),
                        reference=1 * u.TeV)

                sky_model = SkyModel(spectral_model=spectral_model)
                dataset = SpectrumDataset(aeff=aeff, edisp=edisp,
                                          models=sky_model, livetime=livetime)

                counts[ix, iy] = dataset.npred().data.sum()

        self.counts = counts


    def integrate_PL(self, emin, emax):
        """
        Integrate the PowerLaw model from emin to emax

        Parameters
        ----------------------
        emin : float
            The minimum energy for the integral
        emax : float
            The maximum energy for the integral

        """

        func_min = self.amplitude_PL / (1 - self.index_PL) * emin ** (1 - self.index_PL)
        func_max = self.amplitude_PL / (1 - self.index_PL) * emax ** (1 - self.index_PL)

        integral = func_max - func_min

        return integral

    def sum_region(self, coords, radius):
        """
        Calculate the energy and flux in a given region

        Parameters
        ----------
        coords : array
            The galactic coordinates for the summation
        radius : float
            The radius of the summing region
        """

        wcs = WCS(self.header, naxis=2)
        position = SkyCoord(l=coords[0], b=coords[1], frame='galactic')
        aperture = SkyCircularAperture(position, radius).to_pixel(wcs)

        mask = aperture.to_mask(method='center')

        fluxes = np.empty((self.data.shape[0]))

        for i in range(self.data.shape[0]):
            fluxes[i] = np.sum(mask.multiply(self.data[i, :, :]))

        energies = np.logspace(self.header['CRVAL3'],
                               self.header['CRVAL3'] + (self.header['NAXIS3'] - 1) * self.header['CDELT3'],
                               self.header['NAXIS3'])

        return energies, fluxes


    def pixel(self, coords):
        """
        Calculate the energy and flux for a given coordinate

        Parameters
        ----------
        coords : array
            The galactic coordinates for the spot of interest
        """
        wcs = WCS(self.header, naxis=2)
        coord = SkyCoord(l=coords[0], b=coords[1], frame='galactic')
        lpixel, bpixel = wcs.world_to_pixel(coord)

        fluxes = self.data[:, int(bpixel), int(lpixel)]
        energies = np.logspace(self.header['CRVAL3'],
                               self.header['CRVAL3'] + (self.header['NAXIS3'] - 1) * self.header['CDELT3'],
                               self.header['NAXIS3'])

        return energies, fluxes
