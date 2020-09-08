import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import SkyCoord
from mario.utils.model_utils import calc_Bmag, calc_dist_CR_release_point, escape_time
from astropy.io import fits


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

        self.tsedov = tsedov
        self.Ec = Ec
        self.Ec_max = Ec_max
        self.deltap = deltap
        self.n0 = n0
        
        self.E51 = 1
        self.u1 = 1.4 * u.kg

        self.Ebudget = Ebudget

        self.model = 'ProtonSpectrum'


    def time(self, Ep):
        return self.snr.age

    def calc_release_point(self, Ec=None):
        if Ec is None:
            Ec = self.Ec
        tesc = escape_time(tsedov=self.tsedov, Ec=Ec, Ec_max=self.Ec_max, delta=self.deltap)
        return calc_dist_CR_release_point(E51=self.E51, n=self.n0, u1=self.u1, tesc=tesc)

    @u.quantity_input(Ep=u.GeV, R=u.pc, Rc=u.pc, B_mag=u.uG)
    def flux(self, Ep, R, Rc=None, B_mag=None):
        """
        Calculates the proton flux of your counterpart

        Parameters
        ----------------------
        Ep : array
            The energy of cosmic ray protons
        R : float
            The distance from the source centre to the cloud
        Rc : float, optional
            The distance from the source centre to the release point of cosmic ray protons
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
            if len(np.shape(R.value)) == 2:
                R = R * np.ones((len(Ep), np.shape(R)[0], np.shape(R)[1]))

        if Rc is None:
            Rc = 0*u.pc

        if B_mag is None:
            B_mag = calc_Bmag(self.nH)

        # Calculation of diffusion coefficient
        D0 = 3e27 * u.cm ** 2 / u.s
        diff_coeff = self.diff_chi * D0 * (Ep / u.GeV / (B_mag / (3 * u.uG))) ** self.diff_delta

        # Calculation of cooling time of protons
        cool_time = (6e7 * (self.nH / u.cm ** -3) ** (-1) * u.yr).to(u.s)

        # Calculation of diffusion radius
        frac = ((np.exp(self.time(Ep) * self.diff_delta / cool_time) - 1)
                / (self.time(Ep) * self.diff_delta / cool_time))
        diff_radius = 2 * np.sqrt(diff_coeff * self.time(Ep) * frac)
               
        # Calculation of normalisation constant, N0
        alpha = 2
        E_lower = (1 * u.GeV).value
        E_upper = (1e8 * u.GeV).value
        intt = (E_upper**(1-alpha) / (1-alpha)) - (E_lower**(1-alpha) / (1-alpha))
        N0 = self.Ebudget.to(u.GeV) / ( intt * u.GeV)

        # Calculation of proton flux
        coeff = (N0 * (Ep / u.GeV) ** (-alpha)) / (np.pi ** (3 / 2) * diff_radius ** 3)
        expo = (- ((alpha - 1) * self.snr.age / cool_time) - ((R-Rc) / diff_radius) ** 2)
        flux = (coeff / u.GeV * np.exp(expo)).to(u.TeV ** (-1) * u.cm ** (-3))

        flux[(R-Rc) <= 0 ] = 0 * (u.TeV ** (-1) * u.cm ** (-3))
        flux[self.time(Ep) <= 0 ] = 0 * (u.TeV ** (-1) * u.cm ** (-3))

        self.energies = Ep
        self.fluxes = flux

        return flux

    def save_cube(self, gp, author='', filename='ProtonFlux_cube.fits'):
        """
        Save the proton flux to a fits cube

        Parameters
        ----------------------
        author : str, optional
            Author of the fits file (default 'Kirsty Feijen')
        filename : str, optional
            The name to save your fits to (default 'Protonflux_cube.fits')

        """
        step = np.diff(np.log10(self.energies[:,0,0].to(u.TeV).value))[0]

        E0 = np.log10(self.energies[0, 0, 0].to(u.TeV).value)

        hdr = fits.PrimaryHDU(data=self.fluxes.value)
        hdr.header['BUNIT'] = f'{self.fluxes.unit}'
        
        hdr.header['CTYPE1'] = gp.header['CTYPE1']
        hdr.header['CRVAL1'] = gp.header['CRVAL1']
        hdr.header['CRPIX1'] = gp.header['CRPIX1']
        hdr.header['CDELT1'] = gp.header['CDELT1']

        hdr.header['CTYPE2'] = gp.header['CTYPE2']
        hdr.header['CRVAL2'] = gp.header['CRVAL2']
        hdr.header['CRPIX2'] = gp.header['CRPIX2']
        hdr.header['CDELT2'] = gp.header['CDELT2']

        hdr.header['CTYPE3'] = 'log ENERGY'
        hdr.header['CUNIT3'] = 'TeV'
        hdr.header['CRVAL3'] = E0
        hdr.header['CRPIX3'] = 1.
        hdr.header['CDELT3'] = step
        
        hdr.header['GDELTA'] = f'{self.diff_delta}'
        hdr.header['GCHI'] = f'{self.diff_chi}'
        hdr.header['GEBUDGET'] = f'{self.Ebudget}'

        hdr.header['GAGE'] = f'{self.snr.age}'

        hdr.header['GMODEL'] = 'Hadronic'

        hdr.header['COMMENT'] = f'tsedov={self.tsedov}, Ec_max={self.Ec_max}'
        hdr.header['COMMENT'] = f'Ec={self.Ec}'
        hdr.header['COMMENT'] = f'n0={self.n0}, deltap={self.deltap}' 

        hdr.header['AUTHOR'] = author
        hdr.writeto(filename, overwrite=True)


class ProtonSpectrumShell:
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

        self.tsedov = tsedov
        self.Ec = Ec
        self.Ec_max = Ec_max
        self.deltap = deltap
        self.n0 = n0
        
        self.E51 = 1
        self.u1 = 1.4 * u.kg

        self.Ebudget = Ebudget

        self.model = 'ProtonSpectrumShell'

    def time(self):
        tesc = escape_time(tsedov=self.tsedov, Ec=self.Ec, Ec_max=self.Ec_max, delta=self.deltap)
        time_esc = (self.snr.age - tesc)
        return time_esc

    @u.quantity_input(Ep=u.GeV, R=u.pc, Rc=u.pc, B_mag=u.uG)
    def flux(self, Ep, R, Rc=None, B_mag=None):
        """
        Calculates the proton flux of your counterpart

        Parameters
        ----------------------
        Ep : array
            The energy of cosmic ray protons
        R : float
            The distance from the source centre to the cloud
        Rc : float, optional
            The distance from the source centre to the release point of cosmic ray protons
        B_mag : float, optional
            The magentic field of the region surrounding the counterpart

        Returns
        ----------------------
        array
            An array of values for proton-flux

        Examples
        ----------------------
        >>> Ep = np.logspace(0, 6, 50)*u.GeV
        >>> R = 12*u.pc
        >>> ProtonSpectrum.flux(Ep, R)

        """
        if Rc is None:
            Rc = 0*u.pc

        if type(Ep.value) == np.ndarray:
            Ep = Ep[:, None, None]
            if len(np.shape(R.value)) == 2:
                R = R * np.ones((len(Ep), np.shape(R)[0], np.shape(R)[1]))

        if B_mag is None:
            B_mag = calc_Bmag(self.nH)

        # Calculation of diffusion coefficient
        D0 = 3e27 * u.cm ** 2 / u.s
        diff_coeff = self.diff_chi * D0 * (Ep / u.GeV / (B_mag / (3 * u.uG))) ** self.diff_delta

        # Calculation of cooling time of protons
        cool_time = (6e7 * (self.nH / u.cm ** -3) ** (-1) * u.yr).to(u.s)

        # Calculation of diffusion radius
        frac = ((np.exp(self.time() * self.diff_delta / cool_time) - 1)
                / (self.time() * self.diff_delta / cool_time))
        diff_R = 2 * np.sqrt(diff_coeff * self.time() * frac)
               
        # Calculation of normalisation constant, N0
        alpha = 2
        E_lower = (1 * u.GeV).value
        E_upper = (1e8 * u.GeV).value
        intt = (E_upper**(1-alpha) / (1-alpha)) - (E_lower**(1-alpha) / (1-alpha))
        N0 = self.Ebudget.to(u.GeV) / ( intt * u.GeV)

        # Calculation of proton flux
        coeff = (N0 * (Ep / u.GeV) ** (-alpha)) / (np.pi ** (3 / 2) * ((Rc + diff_R)** 3 - Rc**3))
        expo = (- ((alpha - 1) * self.time() / cool_time) - ((R-Rc)**2 / (diff_R**2)))
        flux = (coeff / u.GeV * np.exp(expo)).to(u.TeV ** (-1) * u.cm ** (-3))

        flux[(R-Rc) <= 0 ] = 0 * (u.TeV ** (-1) * u.cm ** (-3))
        flux[self.time() <= 0 ] = 0 * (u.TeV ** (-1) * u.cm ** (-3))

        self.energies = Ep
        self.fluxes = flux

        return flux

    def save_cube(self, gp, author='', filename='ProtonFlux_cube.fits'):
        """
        Save the proton flux to a fits cube

        Parameters
        ----------------------
        author : str, optional
            Author of the fits file (default 'Kirsty Feijen')
        filename : str, optional
            The name to save your fits to (default 'Protonflux_cube.fits')

        """
        step = np.diff(np.log10(self.energies[:,0,0].to(u.TeV).value))[0]

        E0 = np.log10(self.energies[0, 0, 0].to(u.TeV).value)

        hdr = fits.PrimaryHDU(data=self.fluxes.value)
        hdr.header['BUNIT'] = f'{self.fluxes.unit}'
        
        hdr.header['CTYPE1'] = gp.header['CTYPE1']
        hdr.header['CRVAL1'] = gp.header['CRVAL1']
        hdr.header['CRPIX1'] = gp.header['CRPIX1']
        hdr.header['CDELT1'] = gp.header['CDELT1']

        hdr.header['CTYPE2'] = gp.header['CTYPE2']
        hdr.header['CRVAL2'] = gp.header['CRVAL2']
        hdr.header['CRPIX2'] = gp.header['CRPIX2']
        hdr.header['CDELT2'] = gp.header['CDELT2']

        hdr.header['CTYPE3'] = 'log ENERGY'
        hdr.header['CUNIT3'] = 'TeV'
        hdr.header['CRVAL3'] = E0
        hdr.header['CRPIX3'] = 1.
        hdr.header['CDELT3'] = step
        
        hdr.header['GDELTA'] = f'{self.diff_delta}'
        hdr.header['GCHI'] = f'{self.diff_chi}'
        hdr.header['GEBUDGET'] = f'{self.Ebudget}'

        hdr.header['GAGE'] = f'{self.snr.age}'

        hdr.header['GMODEL'] = 'Hadronic'

        hdr.header['COMMENT'] = f'tsedov={self.tsedov}, Ec_max={self.Ec_max}'
        hdr.header['COMMENT'] = f'Ec={self.Ec}'
        hdr.header['COMMENT'] = f'n0={self.n0}, deltap={self.deltap}' 

        hdr.header['AUTHOR'] = author
        hdr.writeto(filename, overwrite=True)



class TimeDependentProtonSpectrum(ProtonSpectrumShell):

    def __init__(self, gamma_src, snr, nH, diff_chi=0.01, diff_delta=0.5,
                 tsedov=100 * u.yr, Ec=150 * u.TeV, Ec_max=500 * u.TeV, deltap=1.0,
                 Ebudget=1e50 * u.erg, n0=0.03 * u.cm ** (-3)):
        super(TimeDependentProtonSpectrum, self).__init__(gamma_src, snr, nH, diff_chi, diff_delta,
                                                          tsedov, Ec, Ec_max, deltap, n0, Ebudget)

        self.model = 'TimeDependentProtonSpectrum'
        
    def time(self, Ep):
        tesc = escape_time(tsedov=self.tsedov, Ec=Ep.to(u.TeV), Ec_max=self.Ec_max, delta=self.deltap)
        time_esc = (self.snr.age - tesc)
        return time_esc

    def escape_radius(self, Ep):
        tesc = escape_time(tsedov=self.tsedov, Ec=Ep.to(u.TeV), Ec_max=self.Ec_max, delta=self.deltap)
        return calc_dist_CR_release_point(E51=self.E51, n=self.n0, u1=self.u1, tesc=self.tesc)

    def escape_energy(self):
        # Energy of gamma-rays at the radius of the SNR
        r_snr = self.snr.distance.to(u.pc) * np.tan(self.snr.radius)
        tesc_snr = ( r_snr / (0.31*u.pc * self.E51**(1/5) * (self.n0/u.cm**(-3))**(-1/5) * 
                              (self.u1/ (1.4*u.kg))**(-1/5)))**(5/2) * u.yr
        Ep_esc = self.Ec_max * (tesc_snr / self.tsedov)**(-self.deltap)

        self.Ep_esc = Ep_esc
        self.Eg_esc = Ep_esc * 0.17
        self.tesc_snr = tesc_snr

        return 

class BackgroundProtonSpectrum:
    """
    A class used to determine the background cosmic ray proton spectra

    """

    def __init__(self, gas_map_path):
        
        file = fits.open(gas_map_path)
        self.data = file[0].data * u.cm ** (-2)

        if len(np.shape(self.data)) == 3:
            self.data = file[0].data[0][:][:] * u.cm ** (-2)
        
    def flux(self, Ep):
        
        ysize, xsize =  self.data.shape
        
        f_bg = (2.2 * (Ep.to(u.GeV) / u.GeV) **(-2.75))* (1 / (u.cm**2 * u.GeV * u.sr * u.s) )
        J_bg = (f_bg * (4*np.pi*u.sr / (c.to(u.cm / u.s)))) .to(u.TeV ** (-1) * u.cm ** (-3))

        J_bg_array = (np.ones([Ep.size, ysize, xsize]) * J_bg[:, None, None])

        self.energies = Ep
        self.fluxes = J_bg_array

        return J_bg_array
        

    def save_cube(self, gp, author='', filename='Protonflux_cube.fits'):
        """
        Save the proton flux to a fits cube

        Parameters
        ----------------------
        author : str, optional
            Author of the fits file (default 'Kirsty Feijen')
        filename : str, optional
            The name to save your fits to (default 'Protonflux_cube.fits')

        """
        step = np.diff(np.log10(self.energies.value))[0]

        E0 = np.log10(self.energies[0].value)

        hdr = fits.PrimaryHDU(data=self.fluxes.value)
        hdr.header['BUNIT'] = f'{self.fluxes.unit}'
        
        hdr.header['CTYPE1'] = gp.header['CTYPE1']
        hdr.header['CRVAL1'] = gp.header['CRVAL1']
        hdr.header['CRPIX1'] = gp.header['CRPIX1']
        hdr.header['CDELT1'] = gp.header['CDELT1']

        hdr.header['CTYPE2'] = gp.header['CTYPE2']
        hdr.header['CRVAL2'] = gp.header['CRVAL2']
        hdr.header['CRPIX2'] = gp.header['CRPIX2']
        hdr.header['CDELT2'] = gp.header['CDELT2']

        hdr.header['CTYPE3'] = 'log ENERGY'
        hdr.header['CUNIT3'] = 'TeV'
        hdr.header['CRVAL3'] = E0
        hdr.header['CRPIX3'] = 1.
        hdr.header['CDELT3'] = step

        hdr.header['AUTHOR'] = author
        hdr.writeto(filename, overwrite=True)

