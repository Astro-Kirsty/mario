import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import coordinates as coord
from scipy.optimize import curve_fit
from scipy import integrate
from photutils import SkyCircularAperture
from astropy.wcs import WCS
import matplotlib.pyplot as plt

def oversample(data, header, grid_size=0.005 * u.deg, radius=0.06 * u.deg, model_vals=True):
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
    radius : float, optional
        Size of aperture you want to smooth (default is 0.06deg)
	model_vals : bool, optional
		If the fits file header contains the diffusion parameters (default is True)

    """
    wcs = WCS(header)

    glon_last, glat_first = wcs.all_pix2world(0, 0, 0)
    glon_first, glat_last = wcs.all_pix2world(data.shape[1], data.shape[0], 1)
    full_glon = np.arange(glon_last, glon_first, -grid_size.to(u.deg) / u.deg) * u.deg
    full_glat = np.arange(glat_first, glat_last, grid_size.to(u.deg) / u.deg) * u.deg

    l, b = np.meshgrid(full_glon, full_glat)
    position = SkyCoord(l=l, b=b, frame='galactic').flatten()
    aperture = SkyCircularAperture(position, radius).to_pixel(wcs)
    mask = aperture.to_mask(method='exact')

    data_oversample = []
    for m in range(len(mask)):
        area_non_nan = len(mask[m].multiply(data)[np.logical_not(np.isnan(mask[m].multiply(data)))])
        mask[m].multiply(data)[np.isnan(mask[m].multiply(data))] = np.nansum(mask[m].multiply(data)) / area_non_nan
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

    return hdu.header, data_oversample


def PL_log(E, amplitude, index):
    """
    Powerlaw fit for the gamma-ray flux array
    log(y) = log(a) - b*log(x)

    Parameters
    ----------------------
    E : array
        The array of energies to fit your data to (should be logged)
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
    
def ECO_log(E, amplitude, index, Ecut):
    """
    Exponential cut-off fit for the gamma-ray flux array
    log(y) = log(a * E^(-index) * exp((E0 - E) / Ec ))

    Parameters
    ----------------------
    E : array
        The array of energies to fit your data to
    amplitude : float
        The amplitude of the powerlaw fit
    index : float
        The index of the powerlaw fit
    Ec : float
        The energy cut-off
    """
    E0 = 1*u.TeV
    return np.log10(amplitude * (E / E0.value) **(-index) * np.exp((E0.value - E) / Ecut))
    
    
def BPL_log(E, amplitude, index1, index2, Ebreak):
    """
    Smooth broken power law fit for the gamma-ray flux array
    log(y) = log(a * (E/E0)^(-index1) * [1 + (E/Ebreak)^((index2-index1)/beta)]^-beta

    Parameters
    ----------------------
    E : array
        The array of energies to fit your data to
    amplitude : float
        The amplitude of the powerlaw fit
    index1 : float
        The index of the powerlaw fit
    index2 : float
        The index of the powerlaw fit
    Ebreak :  float
        The energy cut-off
    """
    E0 = 1*u.TeV
    beta = 1
    return np.log10(amplitude * (E / E0.value)**(-index1) * (1 + (E / Ebreak)**((index2-index1)/beta))**(-beta))

    

class SpectralModel:
    def __init__(self, energies, fluxes):
        self.energies = energies
        self.fluxes = fluxes
        
    def fit(self, fit='best'):
        """
        Fit the spectral model

        Parameters
        ----------------------
        fit : str
            Choose which function to fit (default is 'best')
            	Options are 'PL', 'LP', 'ECO'
        """

        if fit == 'best':
            self._fitPL()
            self._fitLP()
            self._fitECO()
            self._fitBPL()
            self.best_fit = np.argmin([self.rsd_dof_LP, 
                                       self.rsd_dof_PL, 
                                       self.rsd_dof_ECO, 
                                       self.rsd_dof_BPL], 
                                      axis=0)
            if self.best_fit==0:
                self.fit_type = 'PL'
            if self.best_fit==1:
                self.fit_type = 'LP'
            if  self.best_fit==2:
                self.fit_type = 'ECO'
            if  self.best_fit==3:
                self.fit_type = 'BPL'
        if fit == 'PL':
            self._fitPL()
            self.fit_type = 'PL'
        if fit == 'LP':
            self._fitLP()
            self.fit_type = 'LP'
        if fit == 'ECO':
            self._fitECO()
            self.fit_type = 'ECO'
        if fit == 'BPL':
            self._fitBPL()
            self.fit_type = 'BPL'

    def _fitPL(self):
        """ Fit the data with a simple powerlaw """
        log_fluxes = np.log10(self.fluxes)
        log_energies = np.log10(self.energies)
        i_1TeV = np.argmin(abs(self.energies - 1))

        popt, pcov = curve_fit(PL_log, log_energies, log_fluxes, p0=[log_fluxes[i_1TeV], 3], maxfev=5000)
        fit_data = 10 ** (PL_log(log_energies, *popt))
        rsd_dof_PL = self._rsd_dof(self.fluxes, fit_data, len(self.fluxes)-len(popt))

        self.popt = popt
        self.amplitude_PL = 10 ** popt[0]
        self.index_PL = popt[1]
        self.rsd_dof_PL = rsd_dof_PL
        self.fit_data = fit_data
        
        
    def _fitLP(self):
        """ Fit the data with a log parabola """
        log_fluxes = np.log10(self.fluxes)
        log_energies = np.log10(self.energies)
        i_1TeV = np.argmin(abs(self.energies - 1))
        
        popt, pcov = curve_fit(LP_log, log_energies, log_fluxes, p0=[log_fluxes[i_1TeV], 3, 0], maxfev=5000)
        fit_data = 10 ** (LP_log(log_energies, *popt))
        rsd_dof_LP = self._rsd_dof(self.fluxes, fit_data, len(self.fluxes)-len(popt))
        
        self.popt = popt
        self.amplitude_LP = 10 ** popt[0]
        self.alpha_LP = popt[1]
        self.beta_LP = popt[2]
        self.rsd_dof_LP = rsd_dof_LP
        self.fit_data = fit_data
        
    def _fitECO(self):
        """ Fit the data with an exponential cutoff """
        i_1TeV = np.argmin(abs(self.energies - 1))
        i_01TeV = np.argmin(abs(self.energies - 0.1))
        
        index_guess = - np.log(self.fluxes[i_01TeV] / self.fluxes[i_1TeV]) / (np.log(0.1))
        popt, pcov = curve_fit(ECO_log, self.energies, np.log10(self.fluxes),
                               p0=[self.fluxes[i_1TeV], index_guess, 10], 
                               maxfev=5000,
                               bounds=([[self.fluxes[i_1TeV]*0.95, index_guess*0.95, 10], 
                                       [self.fluxes[i_1TeV]*1.05, index_guess*1.05, 100]]),
                               sigma=np.sqrt(self.fluxes))
        fit_data = 10 ** (ECO_log(self.energies, *popt))
        rsd_dof_ECO = self._rsd_dof(self.fluxes, fit_data, len(self.fluxes)-len(popt))

        self.popt = popt
        self.amplitude_ECO = popt[0]
        self.index_ECO = popt[1]
        self.Ecut_ECO = popt[2]
        self.rsd_dof_ECO = rsd_dof_ECO
        self.fit_data = fit_data
        
    def _fitBPL(self):
        """ Fit the data with an smooth broken powerlaw """
        i_1TeV = np.argmin(abs(self.energies - 1))        
        log_fluxes = np.log10(self.fluxes)
        popt, pcov = curve_fit(BPL_log, self.energies, log_fluxes,
                           	p0=[self.fluxes[i_1TeV], 2, 2.6, 0.2], 
                           	maxfev=7500)
        amplitude_BPL = popt[0]
        index1_BPL = popt[1]
        index2_BPL = popt[2]
        Ebreak_BPL = popt[3]
        fit_data = 10 ** (BPL_log(self.energies, *popt))
        rsd_dof_BPL = self._rsd_dof(self.fluxes, fit_data, len(self.fluxes)-len(popt))
            
        self.popt = popt
        self.amplitude_BPL = amplitude_BPL
        self.index1_BPL = index1_BPL
        self.index2_BPL = index2_BPL
        self.Ebreak_BPL = Ebreak_BPL
        self.rsd_dof_BPL = rsd_dof_BPL
        self.fit_data = fit_data
      
    def _rsd_dof(self, model, model_fit, dof):
        """
        Calculates the sum of relative square distance divided by the model fit all divided by the 
        degrees of freedom

        Parameters
        ----------------------
        model : array
            The model array
        model_fit : array
            The fit to the model
        dof : int
            The number of degrees of freedom in the fit

        """

        rsd = np.sum((model - model_fit) ** 2 / model)
        return rsd / dof
    
    def plot(self, fig=None, ax=None, **kwargs):
        """
        Plot the spectral model

        Parameters
        ----------------------
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

        ax.plot(self.energies, self.fluxes, **kwargs)

        ax.set_xlabel('Energy (TeV)')
        ax.set_ylabel('$\gamma$-ray flux (TeV$^{-1}$cm$^{-2}$s$^{-1}$)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='lower left', labelspacing=0.3)

        return fig, ax

    def plot_E2F(self, yerr=None, xerr=None, fig=None, ax=None, **kwargs):
        """
        Plot the spectral model

        Parameters
        ----------------------
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

        ax.errorbar(self.energies, self.energies**2 * self.fluxes, yerr, xerr, **kwargs)

        ax.set_xlabel('Energy (TeV)')
        ax.set_ylabel('$\gamma$-ray energy flux (TeV cm$^{-2}$s$^{-1}$)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='lower left', labelspacing=0.3)

        return fig, ax

    def plot_errorbar(self, yerr=None, xerr=None, fig=None, ax=None, **kwargs):
        """
        Plot the spectral model with errorbars

        Parameters
        ----------------------
        fig : str
            Set the figure for the plot
        ax : str
            Set the axis for the figure
        yerr : float
            Error for the y-axis
        xerr : float
            Error for the x-axis
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
        ax.legend()

        return fig, ax  
        
       
    def plot_butterfly(self, E, F, dF, fig=None, ax=None, **kwargs):
        """
        Plot the butterfly for a spectral model

        Parameters
        ----------------------
        fig : str
            Set the figure for the plot
        ax : str
            Set the axis for the figure
        E : array
            Energy array for butterfly
        F : array
            Flux array for butterfly
        dF : array
            Absolute error in flux array for butterfly
        kwargs : keywords
            Initialisation keywords
        """
        if fig is None:
            fig = plt.figure(figsize=(6, 4))

        if ax is None:
            ax = fig.add_subplot(111)
            
        ax.fill_between(E.to(u.TeV), 
                 (F+dF).to(1 / (u.TeV * u.cm**2 * u.s)), 
                 (F-dF).to(1 / (u.TeV * u.cm**2 * u.s)), 
                 **kwargs)
        return fig, ax 
    


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
            chi = self.header['HCHI']
            delta = self.header['HDELTA']
            Ebudget = self.header['HEB_ERG']

            ax.text(0.03, 0.9, 
                    f'$\chi$={chi}, $\delta$={delta}, '+'E$_{budget}$='+f'{float(Ebudget):.1e}erg',
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    fontsize=15, color='black')

        return fig, ax

    def calc_residual(self, obs_data, obs_header, gamsrc_glon, gamsrc_glat, gamsrc_radius):
        """
        Calculate the residual (observed_data - predicted_data)

        Parameters
        ----------------------
        obs_data : array
            The observation fits file data 
        obs_header : str
            The header of the observation fits file
        gamsrc_glon : float
            The galactic longitude of the gamma-ray source
        gamsrc_glat : float
            The galactic latitude of the gamma-ray source
        gamsrc_radius : float
            The radius of the gamma-ray source

        """

        model_wcs = WCS(self.header)
        glon_last, glat_first = model_wcs.all_pix2world(0, 0, 0)
        glon_first, glat_last = model_wcs.all_pix2world(self.data.shape[1], self.data.shape[0], 1) 

        obs_wcs = WCS(obs_header)

        last_lon, last_lat = coord.SkyCoord(l=glon_last*u.deg, b=glat_last*u.deg, 
                                            frame='galactic').to_pixel(obs_wcs)
        first_lon, first_lat = coord.SkyCoord(l=glon_first*u.deg, b=glat_first*u.deg, 
                                              frame='galactic').to_pixel(obs_wcs)

        obs_crop = obs_data[int(first_lat):int(last_lat), int(last_lon):int(first_lon)]

        position = SkyCoord(l=gamsrc_glon, b=gamsrc_glat, frame='galactic')
        aperture = SkyCircularAperture(position, gamsrc_radius).to_pixel(model_wcs)
        mask = aperture.to_mask(method='center')

        obs_aperture = mask.multiply(obs_crop)
        model_aperture = mask.multiply(self.data)
        
        obs_aperture[mask.data == 0] = np.nan
        model_aperture[mask.data == 0] = np.nan
        
        self.obs_aperture = obs_aperture
        self.model_aperture = model_aperture

        residuals = obs_aperture - model_aperture

        hdu = fits.PrimaryHDU(data=residuals)
        hdu.header['NAXIS1'] = residuals.shape[1]
        hdu.header['NAXIS2'] = residuals.shape[0]
        hdu.header['CTYPE1'] = self.header['CTYPE1']
        hdu.header['CRVAL1'] = (gamsrc_glon + gamsrc_radius).value
        hdu.header['CRPIX1'] = 0.0

        hdu.header['CDELT1'] = self.header['CDELT1']
        hdu.header['CTYPE2'] = self.header['CTYPE2']
        hdu.header['CRVAL2'] = (gamsrc_glat - gamsrc_radius).value
        hdu.header['CRPIX2'] = 0.0
        hdu.header['CDELT2'] = self.header['CDELT2']

        return hdu.header, residuals


class SpatialModel3D(SpatialModel):
    """
    A class for simulating gamma-ray counts
    """

    def __init__(self, data, header, energies=None):
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
        >>> sm = SpatialModel3D(data, header)
        >>> sm.fit('ECO')
        >>> integral = sm.integrate(Emin=1*u.TeV, Emax=100*u.TeV, fit='ECO')

        """
        self.energies = energies
        if energies is None:
	        self.energies = np.logspace(self.header['CRVAL3'],
                                    	self.header['CRVAL3'] + (self.header['NAXIS3'] - 1) * self.header['CDELT3'],
                                    	self.header['NAXIS3'])


    def fit(self, fit='ECO'):
        """ 
        Choosing the fit for the data 

        Parameters
        ----------------------
        fit : str (options 'best', 'PL', 'LP', 'ECO', 'BPL')
            The fit for the data
        """

        if fit == 'best':
            self._fitPL()
            self._fitLP()
            self._fitECO()
            self._fitBPL()
            self.best_fit = np.argmin([self.rsd_dof_LP, 
                                       self.rsd_dof_PL, 
                                       self.rsd_dof_ECO, 
                                       self.rsd_dof_BPL], 
                                       axis=0)
            self.fit_type = 'best'
        if fit == 'PL':
            self._fitPL()
            self.fit_type = 'PL'
        if fit == 'LP':
            self._fitLP()
            self.fit_type = 'LP'
        if fit == 'ECO':
            self._fitECO()
            self.fit_type = 'ECO'
        if fit == 'BPL':
            self._fitBPL()
            self.fit_type = 'BPL'

    def _fitPL(self):
        """ Fit the data with a simple powerlaw """
        Esize, xsize, ysize = self.data.shape
        amplitude_PL = np.empty((xsize, ysize))
        index_PL = np.empty((xsize, ysize))
        rsd_dof_PL = np.empty((xsize, ysize))

        log_energies = np.log10(self.energies)
        data = self.data
        i_1TeV = np.argmin(abs(self.energies - 1))
        i_01TeV = np.argmin(abs(self.energies - 0.1))

        for ix in range(xsize):
            for iy in range(ysize):
                fluxes = data[:, ix, iy]
                log_fluxes = np.log10(fluxes)
                if np.isfinite(log_fluxes).all():
                    index_guess = - np.log(fluxes[i_01TeV] / fluxes[i_1TeV]) / (np.log(0.1))
                    if np.isnan(index_guess)==True:
                        index_guess = 1                
                    popt, pcov = curve_fit(PL_log,
                                           log_energies, log_fluxes,
                                           p0=[log_fluxes[i_1TeV], index_guess], maxfev=5000)
                    amplitude_PL[ix, iy] = 10 ** popt[0]
                    index_PL[ix, iy] = popt[1]
                    rsd_dof_PL[ix, iy] = self._rsd_dof(fluxes, 10 ** PL_log(log_energies, *popt), len(fluxes) - len(popt))
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
        i_1TeV = np.argmin(abs(self.energies - 1))

        for ix in range(xsize):
            for iy in range(ysize):
                fluxes = data[:, ix, iy]
                log_fluxes = np.log10(fluxes)
                if np.isfinite(log_fluxes).all():
                    popt, pcov = curve_fit(LP_log,
                                           log_energies, log_fluxes,
                                           p0=[log_fluxes[i_1TeV], 3, 0], maxfev=5000)
                    amplitude_LP[ix, iy] = 10 ** popt[0]
                    alpha_LP[ix, iy] = popt[1]
                    beta_LP[ix, iy] = popt[2]
                    rsd_dof_LP[ix, iy] = self._rsd_dof(fluxes, 10 ** LP_log(log_energies, *popt), len(fluxes) - len(popt))
                else:
                    amplitude_LP[ix, iy] = 0
                    alpha_LP[ix, iy] = 100
                    beta_LP[ix, iy] = 100
                    rsd_dof_LP[ix, iy] = 1000

        self.amplitude_LP = amplitude_LP
        self.alpha_LP = alpha_LP
        self.beta_LP = beta_LP
        self.rsd_dof_LP = rsd_dof_LP

    def _fitECO(self):
        """ Fit the data with an exponential cutoff """
        Esize, xsize, ysize = self.data.shape
        amplitude_ECO = np.empty((xsize, ysize))
        index_ECO = np.empty((xsize, ysize))
        Ecut_ECO = np.empty((xsize, ysize))
        rsd_dof_ECO = np.empty((xsize, ysize))

        data = self.data
        i_1TeV = np.argmin(abs(self.energies - 1))
        i_01TeV = np.argmin(abs(self.energies - 0.1))

        for ix in range(xsize):
            for iy in range(ysize):
                fluxes = data[:, ix, iy]
                log_fluxes = np.log10(fluxes)
                if np.isfinite(fluxes).all()  & np.isfinite(log_fluxes).all():
                    index_guess = - np.log(fluxes[i_01TeV] / fluxes[i_1TeV]) / (np.log(0.1))
                    if np.isnan(index_guess)==True:
                        index_guess = 1
                    popt, pcov = curve_fit(ECO_log,
                                           self.energies, log_fluxes,
                                           p0=[fluxes[i_1TeV], index_guess, 20], 
                                           maxfev=5000,
                                           bounds=([[fluxes[i_1TeV]*0.95, index_guess*0.95, 10], 
                                                   [fluxes[i_1TeV]*1.05, index_guess*1.05, 100]]),
                                           sigma=np.sqrt(fluxes))
                    amplitude_ECO[ix, iy] = popt[0]
                    index_ECO[ix, iy] = popt[1]
                    Ecut_ECO[ix, iy] = popt[2]
                    rsd_dof_ECO[ix, iy] = self._rsd_dof(fluxes, 10 ** ECO_log(self.energies, *popt), len(fluxes) - len(popt))
                else:
                    amplitude_ECO[ix, iy] = 0
                    index_ECO[ix, iy] = 100
                    Ecut_ECO[ix, iy] = 100
                    rsd_dof_ECO[ix, iy] = 1000

        self.amplitude_ECO = amplitude_ECO
        self.index_ECO = index_ECO
        self.Ecut_ECO = Ecut_ECO
        self.rsd_dof_ECO = rsd_dof_ECO

    def _fitBPL(self):
        """ Fit the data with a smooth broken powerlaw """
        Esize, xsize, ysize = self.data.shape
        amplitude_BPL = np.empty((xsize, ysize))
        index1_BPL = np.empty((xsize, ysize))
        index2_BPL = np.empty((xsize, ysize))
        Ebreak_BPL = np.empty((xsize, ysize))
        rsd_dof_BPL = np.empty((xsize, ysize))

        data = self.data
        i_1TeV = np.argmin(abs(self.energies - 1))

        for ix in range(xsize):
            for iy in range(ysize):
                fluxes = data[:, ix, iy]
                log_fluxes = np.log10(fluxes)
                if np.isfinite(fluxes).all()  & np.isfinite(log_fluxes).all():
                    popt, pcov = curve_fit(BPL_log,
                                           self.energies, log_fluxes,
                                           p0=[fluxes[i_1TeV], 2, 2.6, 0.2], 
                                           bounds=([[fluxes[i_1TeV]*0.9, 0.4, 1, 0.001], 
                                                   [fluxes[i_1TeV]*1.1, 5, 6, 150]]),
                                           maxfev=7500)
                    amplitude_BPL[ix, iy] = popt[0]
                    index1_BPL[ix, iy] = popt[1]
                    index2_BPL[ix, iy] = popt[2]
                    Ebreak_BPL[ix, iy] = popt[3]
                    rsd_dof_BPL[ix, iy] = self._rsd_dof(fluxes, 10 ** BPL_log(self.energies, *popt), len(fluxes) - len(popt))
                else:
                    amplitude_BPL[ix, iy] = 0
                    index1_BPL[ix, iy] = 100
                    index2_BPL[ix, iy] = 100
                    Ebreak_BPL[ix, iy] = 100
                    rsd_dof_BPL[ix, iy] = 1000

        self.amplitude_BPL = amplitude_BPL
        self.index1_BPL = index1_BPL
        self.index2_BPL = index2_BPL
        self.Ebreak_BPL = Ebreak_BPL
        self.rsd_dof_BPL = rsd_dof_BPL
        

    def _rsd_dof(self, model, model_fit, dof):
        """
        Calculates the relative square distance divided by the degrees of freedom

        Parameters
        ----------------------
        model : array
            The model array
        model_fit : array
            The fit to the model
        dof : int
            The number of degrees of freedom in the fit

        """
        rsd = np.sum((model - model_fit) ** 2 / model)
        return rsd / dof

    def _integrate_PL(self, Emin=1*u.TeV, Emax=100*u.TeV):
        """
        Integrate the PowerLaw model from Emin to Emax

        Parameters
        ----------------------
        Emin : float
            The minimum energy for the integral
        Emax : float
            The maximum energy for the integral
        """
        int_min = self.amplitude_PL * (Emin/u.TeV) ** (1 - self.index_PL) / (1 - self.index_PL) 
        int_max = self.amplitude_PL * (Emax/u.TeV) ** (1 - self.index_PL) / (1 - self.index_PL)
        integral = (int_max - int_min)
        return integral

    def _integrate_LP(self, Emin=1*u.TeV, Emax=100*u.TeV):
        """
        Integrate the LogParabola model from Emin to Emax

        Parameters
        ----------------------
        Emin : float
            The minimum energy for the integral
        Emax : float
            The maximum energy for the integral
        """
        Esize, xsize, ysize = self.data.shape
        E = self.energies

        integral = np.empty((xsize, ysize))
        for ix in range(xsize):
            for iy in range(ysize):
                LP_linear = lambda E: (self.amplitude_LP[ix, iy] * 
                                       E ** (- self.alpha_LP[ix, iy] - self.beta_LP[ix, iy]*np.log10(E)))
                integral[ix, iy] = integrate.quad(LP_linear, Emin/u.TeV, Emax/u.TeV)[0]

        return integral

    def _integrate_ECO(self, Emin=1*u.TeV, Emax=100*u.TeV):
        """
        Integrate the Exponential Cutoff model from Emin to Emax

        Parameters
        ----------------------
        Emin : float
            The minimum energy for the integral
        Emax : float
            The maximum energy for the integral
        """
        Esize, xsize, ysize = self.data.shape
        E = self.energies

        integral = np.empty((xsize, ysize))
        for ix in range(xsize):
            for iy in range(ysize):
                ECO_linear = lambda E: (self.amplitude_ECO[ix, iy] * E **(-self.index_ECO[ix, iy]) * 
                                        np.exp( (1 - E) / self.Ecut_ECO[ix, iy]))
                integral[ix, iy] = integrate.quad(ECO_linear, Emin/u.TeV, Emax/u.TeV)[0]

        return integral
        
    def _integrate_BPL(self, Emin=1*u.TeV, Emax=100*u.TeV):
        """
        Integrate the smooth broken powerlaw model from Emin to Emax

        Parameters
        ----------------------
        Emin : float
            The minimum energy for the integral
        Emax : float
            The maximum energy for the integral
        """
        Esize, xsize, ysize = self.data.shape
        E = self.energies
        
        beta = 1

        integral = np.empty((xsize, ysize))
        for ix in range(xsize):
            for iy in range(ysize):
                BPL_linear = lambda E: (self.amplitude_BPL[ix, iy] * (E) **(-self.index1_BPL[ix, iy]) * 
                                        (1 + (E/self.Ebreak_BPL[ix,iy])**((self.index2_BPL[ix, iy]-self.index1_BPL[ix, iy])/beta))**(-beta) )
                integral[ix, iy] = integrate.quad(BPL_linear, Emin/u.TeV, Emax/u.TeV)[0]

        return integral
        
    def _integrate_best(self, Emin=1*u.TeV, Emax=100*u.TeV):
        
        xsize, ysize = self.best_fit.shape
        integral = np.empty((xsize, ysize))
        
        for ix in range(xsize):
            for iy in range(ysize):
                if self.best_fit[ix, iy] == 0:
                    int_min = self.amplitude_PL[ix, iy] * (Emin/u.TeV) ** (1 - self.index_PL[ix, iy]) / (1 - self.index_PL[ix, iy]) 
                    int_max = self.amplitude_PL[ix, iy] * (Emax/u.TeV) ** (1 - self.index_PL[ix, iy]) / (1 - self.index_PL[ix, iy])
                    integral[ix, iy] = (int_max - int_min)
                elif self.best_fit[ix, iy] == 1:
                    LP_linear = lambda E: (self.amplitude_LP[ix, iy] * 
                                               E ** (- self.alpha_LP[ix, iy] - self.beta_LP[ix, iy]*np.log10(E)))
                    integral[ix, iy] = integrate.quad(LP_linear, Emin/u.TeV, Emax/u.TeV)[0]
                elif self.best_fit[ix, iy] == 2:
                    ECO_linear = lambda E: (self.amplitude_ECO[ix, iy] * E **(-self.index_ECO[ix, iy]) * 
                                        np.exp( (1 - E) / self.Ecut_ECO[ix, iy]))
                    integral[ix, iy] = integrate.quad(ECO_linear, Emin/u.TeV, Emax/u.TeV)[0]
                elif self.best_fit[ix, iy] == 3:
                    BPL_linear = lambda E: (self.amplitude_BPL[ix, iy] * (E) **(-self.index1_BPL[ix, iy]) * 
                                        (1 + (E/self.Ebreak_BPL[ix,iy])**((self.index2_BPL[ix, iy]-self.index1_BPL[ix, iy])/beta))**(-beta) )
                    integral[ix, iy] = integrate.quad(BPL_linear, Emin/u.TeV, Emax/u.TeV)[0]
                    
        return integral

    def integrate(self, Emin=1*u.TeV, Emax=100*u.TeV, fit='ECO'):
        """
        Use the fit_type to determine which integral function to use

        Parameters
        ----------------------
        Emin : float
            The minimum energy for the integral
        Emax : float
            The maximum energy for the integral
        fit : str (options 'best', 'PL', 'LP', 'ECO')
            The fit for the data
        """
        if (fit == 'PL'):
            integral = self._integrate_PL(Emin, Emax)
        if (fit == 'LP'):
            integral = self._integrate_LP(Emin, Emax)
        if (fit == 'ECO'):
            integral = self._integrate_ECO(Emin, Emax)
        if (fit == 'BPL'):
            integral = self._integrate_BPL(Emin, Emax)
        if (fit == 'best'):
            integral = self._integrate_best(Emin, Emax)

        return integral
    

    def sum_region(self, coords, radius):
        """
        Calculate the energy and flux in a given region

        Parameters
        ----------------------
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
            fluxes[i] = np.nansum(mask.multiply(self.data[i, :, :]))


        return self.energies, fluxes


    def pixel(self, coords):
        """
        Calculate the energy and flux for a given coordinate

        Parameters
        ----------------------
        coords : array
            The galactic coordinates for the spot of interest
        """
        wcs = WCS(self.header, naxis=2)
        coord = SkyCoord(l=coords[0], b=coords[1], frame='galactic')
        lpixel, bpixel = wcs.world_to_pixel(coord)

        self.lpixel = lpixel
        self.bpixel = bpixel

        fluxes = self.data[:, int(bpixel), int(lpixel)]

        return self.energies, fluxes
