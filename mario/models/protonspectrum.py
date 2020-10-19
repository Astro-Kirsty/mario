import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import SkyCoord
from .utilities import calc_Bmag, calc_shock_radius, escape_time
from astropy.io import fits

class Base:
    """
    A class used to determine the cosmic ray proton spectra
    of a counterpart of a gamma-ray sources
    """

    @u.quantity_input(nH=u.cm**(-3))
    def __init__(self,  nH, 
                 diff_chi=0.01, diff_delta=0.5,
                 tsedov=100*u.yr, Emax=500*u.TeV, n0=1*u.cm**(-3), deltap=0.5,
                 Ebudget=1e50*u.erg, alpha=2):
        
        """
        Parameters
        ----------------------
        nH : float
            The number density of the region surrounding the gamma-ray source
        diff_chi : float, optional
            The diffusion suppression factor (default is 0.01)
        diff_delta : float, optional
            The index of diffusion coefficient (default is 0.5)
        tsedov : float, optional
            The time of the Sedov phase (default is 100 yr)
        Emax : float, optional
            Maximum injection energy of cosmic ray protons (default is 500 TeV)
        n0 : float, optional
            Number density at time of cosmic ray release (default is 1 cm^-3)
        Ebudget : float, optional
            Energy budget of cosmic ray protons (default is 1e50 erg)
        alpha : float, optional
            The spectral index of the injection spectra
        """

        self.nH = nH
        self.diff_chi = diff_chi
        self.diff_delta = diff_delta

        self.tsedov = tsedov
        self.Emax = Emax
        self.n0 = n0
        self.deltap = deltap
        
        self.E51 = 1
        self.u1 = 1.4 * u.kg

        self.Ebudget = Ebudget
        self.alpha = alpha  
        self.Emin = 1 * u.GeV
        
        self.N0 = (self.Ebudget * (1-self.alpha) / (((self.Emax/u.GeV)**(1-self.alpha) - (self.Emin/u.GeV)**(1-self.alpha))*u.GeV)).decompose()

class ProtonSpectrum(Base):

    def __init__(self, nH, 
                 diff_chi=0.01, diff_delta=0.5,
                 tsedov=100*u.yr, Emax=500*u.TeV, n0=1*u.cm**(-3), deltap=0.5,
                 Ebudget=1e50*u.erg, alpha=2):
        
        super(ProtonSpectrum, self).__init__(nH, 
                                             diff_chi, diff_delta,
                                             tsedov, Emax, n0, deltap,
                                             Ebudget)
        

    @u.quantity_input(Ep=u.GeV, R=u.pc, B=u.uG, Rc=u.pc)
    def calc_flux(self, Ep, R, t, Rc=0*u.pc, B=None):
        """
        Calculates the proton flux of your counterpart
        Parameters
        ----------------------
        E: array
            The energy of cosmic ray protons
        R : float
            The distance from the centre of the source to the point of interest
        t : float
            The age of the source
        Rc : float
            The distance from the cebtre of the source to the release point of cosmic ray protons
        B_mag : float, optional
            The magentic field of the region surrounding the counterpart
        Returns
        ----------------------
        array
            An array of values for proton-flux
        """
        
        if type(Ep.value) == np.ndarray:
            Ep = Ep[:, None, None]
            if len(np.shape(R.value)) == 2:
                R = R * np.ones((len(Ep), np.shape(R)[0], np.shape(R)[1]))
            elif len(np.shape(R.value)) == 1:
                R = R * np.ones((len(Ep), np.shape(R)[0]))
                R = R[:, :, None]

        Rc = Rc * np.ones(np.shape(R))

        if B is None:
            B = calc_Bmag(self.nH)

        # Calculation of diffusion coefficient
        D0 = 3e27 * u.cm ** 2 / u.s
        diff_coeff = self.diff_chi * D0 * (Ep / u.GeV / (B / (3*u.uG))) ** self.diff_delta

        # Calculation of cooling time of protons
        cool_time = (6e7 * (self.nH / u.cm**-3) ** (-1) * u.yr)

        # Calculation of diffusion length
        frac = ((np.exp(t * self.diff_delta / cool_time) - 1)
                / (t  * self.diff_delta / cool_time))
        Rdif = 2 * np.sqrt(diff_coeff * t  * frac)
               
        # Calculation of proton flux
        coeff = (self.N0 * (Ep / u.GeV) ** (- self.alpha)) / (np.pi ** (3 / 2) * Rdif ** 3)
        f = coeff / u.GeV * np.exp(- (R - Rc)**2 / Rdif**2 ) * np.exp(- ((self.alpha - 1) * t  / cool_time))       

        # Normalise the proton flux
        normalising = np.exp(- ((self.alpha - 1) * t  / cool_time)) * ( 
               (2 * np.sqrt(np.pi) * abs(Rdif) * Rc ** 2)
               + (np.sqrt(np.pi) * Rdif ** 2 * abs(Rdif)) 
               + (4 * Rdif ** 2 * Rc) ) / (np.sqrt(np.pi) * Rdif ** 2 * abs(Rdif))
        
        flux = (f * 1/(normalising.decompose().value)).to(u.TeV ** (-1) * u.cm ** (-3))

        self.energies = Ep
        self.fluxes = flux
        self.time = t


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

        hdr.header['GAGE'] = f'{self.time}'

        hdr.header['GMODEL'] = 'Hadronic'

        hdr.header['COMMENT'] = f'tsedov={self.tsedov}, Ec_max={self.Ec_max}'
        hdr.header['COMMENT'] = f'Ec={self.Ec}'
        hdr.header['COMMENT'] = f'n0={self.n0}, deltap={self.deltap}' 

        hdr.header['AUTHOR'] = author
        hdr.writeto(filename, overwrite=True)


class ProtonSpectrumTime(Base):

    def __init__(self, nH, 
                 diff_chi=0.01, diff_delta=0.5,
                 tsedov=100*u.yr, Emax=500*u.TeV, n0=1*u.cm**(-3), deltap=0.5,
                 Ebudget=1e50*u.erg, alpha=2):
        
        super(ProtonSpectrumTime, self).__init__(nH, 
                                                 diff_chi, diff_delta,
                                                 tsedov, Emax, n0, deltap,
                                                 Ebudget)
        

    @u.quantity_input(Ep=u.GeV, R=u.pc, B=u.uG)
    def calc_flux(self, Ep, R, t, B=None):
        """
        Calculates the time dependent proton flux of your counterpart
        
        Parameters
        ----------------------
        E: array
            The energy of cosmic ray protons
        R : float
            The distance from the centre of the source to the point of interest
        t : float
            The age of the source
        B_mag : float, optional
            The magentic field of the region surrounding the counterpart
        Returns
        ----------------------
        array
            An array of values for proton-flux
        """

        if type(Ep.value) == np.ndarray:
            Ep = Ep[:, None, None]
            if len(np.shape(R.value)) == 2:
                R = R * np.ones((len(Ep), np.shape(R)[0], np.shape(R)[1]))
            elif len(np.shape(R.value)) == 1:
                R = R * np.ones((len(Ep), np.shape(R)[0]))
                R = R[:, :, None]
       
        tesc = escape_time(tsedov=self.tsedov, Ec=Ep.to(u.TeV), Ec_max=self.Emax, deltap=self.deltap)
        Rc = calc_shock_radius(t=tesc, E_SN=self.E_SN, n0=self.n0)
        Rc = Rc * np.ones(np.shape(R))

        time = t - tesc

        if B is None:
            B = calc_Bmag(self.nH)

        # Calculation of diffusion coefficient
        D0 = 3e27 * u.cm ** 2 / u.s
        diff_coeff = self.diff_chi * D0 * (Ep / u.GeV / (B / (3*u.uG))) ** self.diff_delta

        # Calculation of cooling time of protons
        cool_time = (6e7 * (self.nH / u.cm**-3) ** (-1) * u.yr)
        
        # Calculation of diffusion length
        frac = ((np.exp(time * self.diff_delta / cool_time) - 1)
                / (time  * self.diff_delta / cool_time))
        Rdif = 2 * np.sqrt(diff_coeff * time  * frac)
        Rshock = 0.0875 * 4500*u.pc*np.tan(0.375*u.deg)
        Rdif = Rdif + Rshock
               
        # Calculation of proton flux
        coeff = (self.N0 * (Ep / u.GeV) ** (- self.alpha)) / (np.pi ** (3 / 2) * Rdif ** 3)
        f = coeff / u.GeV * np.exp(- (R - Rc)**2 / Rdif**2 ) * np.exp(- ((self.alpha - 1) * time  / cool_time))       

        # Normalise the proton flux
        normalising = np.exp(- ((self.alpha - 1) * time  / cool_time)) * ( 
               (2 * np.sqrt(np.pi) * abs(Rdif) * Rc ** 2)
               + (np.sqrt(np.pi) * Rdif ** 2 * abs(Rdif)) 
               + (4 * Rdif ** 2 * Rc) ) / (np.sqrt(np.pi) * Rdif ** 2 * abs(Rdif))    
        
        flux = (f * 1/(normalising.decompose().value)).to(u.TeV ** (-1) * u.cm ** (-3))

        Ep_esc = (self.Emax * (t / self.tsedov)**(-0.5)).to(u.TeV)
        Rbubble = calc_shock_radius(t=tesc, E_SN=self.E_SN, n0=self.n0)
        Rbubble = Rbubble * np.ones(np.shape(R))       
        
        flux[(R > Rbubble) & (Ep < Ep_esc)] = 0 * u.TeV ** (-1) * u.cm ** (-3)
        flux[(R < Rbubble) & (Ep > Ep_esc)] = 0 * u.TeV ** (-1) * u.cm ** (-3)

        # Filling the bubble
        b_r = 1 / (4/3 * np.pi * Rbubble**3)
        bubble_fill = (self.N0 * ((Ep / u.GeV) ** (- self.alpha)) / u.GeV * b_r).to(u.TeV ** (-1) * u.cm ** (-3))
        flux[(R < Rbubble) & (Ep < Ep_esc)] = bubble_fill[(R < Rbubble) & (Ep < Ep_esc)]

        
        self.energies = Ep
        self.fluxes = flux
        self.time = t
        
        self.Ep_esc = Ep_esc
        self.Rbubble = Rbubble



class ProtonSpectrumShell(Base):

    def __init__(self, nH, 
                 diff_chi=0.01, diff_delta=0.5,
                 tsedov=100*u.yr, Emax=500*u.TeV, n0=1*u.cm**(-3), deltap=0.5,
                 Ebudget=1e50*u.erg, alpha=2):
        
        super(ProtonSpectrumShell, self).__init__(nH, 
                                                  diff_chi, diff_delta,
                                                  tsedov, Emax, n0, deltap,
                                                  Ebudget)
       

    @u.quantity_input(Ep=u.GeV, R=u.pc, B=u.uG, Rc=u.pc)
    def calc_flux(self, Ep, R, t, Rc=0*u.pc, B=None):

        if type(Ep.value) == np.ndarray:
            Ep = Ep[:, None, None]
            if len(np.shape(R.value)) == 2:
                R = R * np.ones((len(Ep), np.shape(R)[0], np.shape(R)[1]))
            elif len(np.shape(R.value)) == 1:
                R = R * np.ones((len(Ep), np.shape(R)[0]))
                R = R[:, :, None]

        Rc = Rc * np.ones(np.shape(R))

        if B is None:
            B = calc_Bmag(self.nH)

        tesc = escape_time(tsedov=self.tsedov, Ec=self.Ec, Ec_max=self.Emax, deltap=self.deltap)
        time = t - tesc

        # Calculation of diffusion coefficient
        D0 = 3e27 * u.cm ** 2 / u.s
        diff_coeff = self.diff_chi * D0 * (Ep / u.GeV / (B / (3*u.uG))) ** self.diff_delta

        # Calculation of cooling time of protons
        cool_time = (6e7 * (self.nH / u.cm**-3) ** (-1) * u.yr)

        # Calculation of diffusion length
        frac = ((np.exp(time * self.diff_delta / cool_time) - 1)
                / (time * self.diff_delta / cool_time))
        Rdif = 2 * np.sqrt(diff_coeff * time  * frac)
               
        # Calculation of proton flux
        coeff = (self.N0 * (Ep / u.GeV) ** (- self.alpha)) / (np.pi ** (3 / 2) * ((Rc + Rdif)** 3 - Rc ** 3))
        f = coeff / u.GeV * np.exp(- (R - Rc)**2 / Rdif**2 ) * np.exp(- ((self.alpha - 1) * time  / cool_time)) 

        # Normalise the proton flux
        normalising = np.exp(- ((self.alpha - 1) * time  / cool_time)) * (
                      (2 * np.sqrt(np.pi) * abs(Rdif) * Rc ** 2) 
                       + (4 * Rdif ** 2 * Rc)
                       + (np.sqrt(np.pi) * Rdif ** 2 * abs(Rdif)))  \
                        / (np.sqrt(np.pi) * ((Rc + Rdif) ** 3 - Rc ** 3))

        flux = (f * 1/(normalising.decompose().value)).to(u.TeV ** (-1) * u.cm ** (-3))
        flux[time <= 0 ] = 0 * (u.TeV ** (-1) * u.cm ** (-3))
        flux[(R-Rc) <= 0 ] = 0 * (u.TeV ** (-1) * u.cm ** (-3))

        self.energies = Ep
        self.fluxes = flux
        self.time = time




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

