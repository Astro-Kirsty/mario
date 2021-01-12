import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.constants import M_sun, m_p

from .utilities import BasePDF
from .utilities import calc_Bmag, calc_shock_radius, calc_energy_evolution
from .utilities import calc_escape_time, calc_escape_energy

class BaseHadronicImpulsivePDF(BasePDF):
    """
    A base class that can be used to create hadronic PDFs
    """

    def __init__(self, n=100*u.cm**-3, n0=1*u.cm**-3,  
                 D0=3e27*u.cm**2/u.s, dif_chi=0.01, dif_delta=0.5, deltap=0.3, 
                 Mej=1.4*u.solMass, Emax_t_ST0=1*u.PeV, 
                 E_SN=1e51*u.erg):
        """
        Parameters
        ----------------------
        n : float, optional
            Average number density in the region surrounding the gamma-ray source (default is 100 cm^-3)
        n0 : float, optional
            Ambient upstream hydrogen number density (default is 1 cm^-3)
        D0 : float, optional
            The galactic diffusion coefficient (default is 3 x 10^27 cm^2/s)
        dif_chi : float, optional
            The diffusion suppression factor (default is 0.01)
        dif_delta : float, optional
            The index of diffusion coefficient (default is 0.5)
        deltap : float, optional 
        	The 
		Mej : float, optional
			Mass of the eject (default is 1.4 Solar Mass for Type Ia)
			For Type II, typically a value of 10 Solar Mass is used
        Emax_t_ST0 : float, optional
            Maximum injection energy of cosmic ray protons at t_ST0 (corresponds to start of 
            Sedov-Taylor phase) (default is 1 PeV)
        E_SN : float, optional
            Kinetic energy of SN explosion (default is 10^51 erg)

        """
        super(BaseHadronicImpulsivePDF, self).__init__(E_SN)
        
        self.n = n
        self.n0 = n0
        self.D0 = D0
        self.dif_chi = dif_chi
        self.dif_delta = dif_delta
        
        self.deltap = deltap
        self.rho = 1 * m_p * u.cm**(-3)
        self.Mej = Mej.to(u.kg)
        self.t_ST0 = 1.6e3*u.yr * (E_SN / (1e51*u.erg))**(-1/2) * (self.Mej / (10*M_sun))**(5/6) * (self.rho / (m_p * u.cm**(-3)))**(-1/3)
        
        self.Emax_t_ST0 = Emax_t_ST0
        
        
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def dif_l(self, E, t):
        self.B = calc_Bmag(self.n)
        # Calculation of diffusion coefficient
        dif_coeff = self.dif_chi * self.D0 * (E / u.GeV / (self.B / (3*u.uG))) ** self.dif_delta
        # Calculation of diffusion length
        dif_l = 2 * np.sqrt(dif_coeff * self.dif_time(E,t))
        return dif_l
    
    @u.quantity_input(E=u.GeV, R=u.pc)
    def calc_pdf(self, E, R, t):
        """
        Calculates the proton probability density function
        Parameters
        ----------------------
        E : array
            The energy of cosmic ray protons
        R : float
            The distance from the centre of the source to the point of interest
        t : float
            The age of the source
        Returns
        ----------------------
        array
            An array of values for the proton pdf
        """
        
        if type(E.value) == np.ndarray:
            E = E[:, None, None]
            if len(np.shape(R.value)) == 2:
                R = R * np.ones((len(E), np.shape(R)[0], np.shape(R)[1]))
            elif len(np.shape(R.value)) == 1:
                R = R * np.ones((len(E), np.shape(R)[0]))
                R = R[:, :, None]
            elif len(np.shape(R.value)) == 0:
                R = R * np.ones((len(E), 1))
                R = R[:, :, None]
        
        Rsh = self.Rsh(E, t) * np.ones(np.shape(R))

        # Define the critical radius and energy in which particles less than these value are trapped
        Rcrit = self.Rcrit(t) * np.ones(np.shape(R))
        self.Resc = Rcrit
        
        Ecrit = calc_escape_energy(t, deltap=self.deltap, t_ST0=self.t_ST0, Emax_t_ST0=self.Emax_t_ST0)
        self.Eesc = Ecrit
        
        # Calculation of diffusion length
        dif_length = self.dif_l(E,t)
        
        pdf = np.zeros(np.shape(R)) / u.cm**3
        
        # Particles with energies less than Ecrit have not escaped and are trapped
        pdf[(E < Ecrit) & (R < Rcrit)] = (1 / (4/3 * np.pi * Rcrit**3))[(E < Ecrit) & (R < Rcrit)]
        
        # Ensure the integral over the distribution function is 1
        norm = ((np.sqrt(np.pi) * dif_length**2 + 2 * np.sqrt(np.pi) * Rsh**2) * np.abs(dif_length) + 4 * Rsh * dif_length**2) / (np.sqrt(np.pi) * dif_length**3)
        
        # Calculation of proton pdf 
        coeff = 1. / (np.pi ** (3 / 2) * dif_length ** 3)
        pdf[(E >= Ecrit) & (R >= Rcrit)] = (coeff * np.exp(- (R - Rsh)**2 / dif_length**2 ) / norm)[(E > Ecrit) & (R > Rcrit)]
        
        self.Emax_t = (self.Emax_t_ST0 * ((t / self.t_ST0).decompose()) ** (-self.deltap)).to(u.TeV)
        self.R = R
        self.t = t
        self.energy = E[:,0,0]
        self.pdf = pdf
        
        
    def save(self, author='', basename='HadronicPDF'):
        """
        Save the hadronic PDF to a fits file
        """
        try:
            filename = f'{basename}_{self.name}_chi{self.dif_chi}_delta{self.dif_delta}_D0{self.D0.to(u.cm**2/u.s).value}_n{self.n.to(u.cm**-3).value}_n0{self.n0.to(u.cm**-3).value}_ESN{self.E_SN.to(u.erg).value}_EmaxST0{self.Emax_t_ST0.to(u.PeV).value}_tST0{self.t_ST0.to(u.yr).value:.2f}_deltap{self.deltap}.fits'

            step = np.diff(np.log10(self.energy[:].to(u.TeV).value))[0]

            E0 = np.log10(self.energy[0].to(u.TeV).value)

            hdr = fits.PrimaryHDU(data=self.pdf.value)
            hdr.header['BUNIT'] = f'{self.pdf.unit}'

            hdr.header['CTYPE1'] = 'Distance SNR'
            hdr.header['CUNIT1'] = 'pc'
            hdr.header['CRVAL1'] = self.R[0,0,0].to(u.pc).value
            hdr.header['CRPIX1'] = 1.
            hdr.header['CDELT1'] = np.diff(self.R[0,:,0].to(u.pc).value)[0]

            hdr.header['CTYPE2'] = 'Distance SNR'
            hdr.header['CUNIT2'] = 'pc'
            hdr.header['CRVAL2'] = self.R[0,0,0].to(u.pc).value
            hdr.header['CRPIX2'] = 1.
            hdr.header['CDELT2'] = np.diff(self.R[0,0,:].to(u.pc).value)[0]

            hdr.header['CTYPE3'] = 'log ENERGY'
            hdr.header['CUNIT3'] = 'TeV'
            hdr.header['CRVAL3'] = E0
            hdr.header['CRPIX3'] = 1.
            hdr.header['CDELT3'] = step
            
            hdr.header['HNAME'] = self.name
            hdr.header['HT_KYR'] = self.t.to(u.kyr).value
            hdr.header['HCHI'] = self.dif_chi
            hdr.header['HDELTA'] = self.dif_delta
            hdr.header['HD0_CM2S'] = self.D0.to(u.cm**2/u.s).value
            hdr.header['HN_CM3'] = self.n.to(u.cm**-3).value
            hdr.header['HN0_CM3'] = self.n0.to(u.cm**-3).value
            hdr.header['HESN_ERG'] = self.E_SN.to(u.erg).value
            hdr.header['HEMT_TEV'] = self.Emax_t.to(u.TeV).value
            hdr.header['HEMT0_P'] = self.Emax_t_ST0.to(u.PeV).value
            hdr.header['HTST0_YR'] = self.t_ST0.to(u.yr).value
            hdr.header['HDELTAP'] = self.deltap
            hdr.header['HEES_TEV'] = self.Eesc.to(u.TeV).value
            hdr.header['HRESC_PC'] = self.Resc[0,0,0].to(u.pc).value

            hdr.header['AUTHOR'] = author

            hdr.writeto(filename, overwrite=True)
        except:
            print('Please calculate the pdf first')
 

class HadronicImpulsivePDF(BaseHadronicImpulsivePDF):

    def __init__(self, n=100/u.cm**3, n0=1/u.cm**3,
                 D0=3e27*u.cm**2/u.s, dif_chi=0.01, dif_delta=0.5, deltap=0.3,
                 Mej=1.4*u.solMass, Emax_t_ST0=1*u.PeV, 
                 E_SN=1e51*u.erg,
                 Rshock=None):
        
        super(HadronicImpulsivePDF, self).__init__(n, n0,
            D0, dif_chi, dif_delta, deltap,
            Mej, Emax_t_ST0, E_SN)
        
        self.Rshock = Rshock
        
        self.name = 'AharonianImpulsive'
        
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def dif_time(self, E, t):
        return t * np.ones(np.shape(E))
    
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def Rsh(self, E, t):
        return self.Rcrit(t=t)
    
    @u.quantity_input(t=u.kyr)
    def Rcrit(self, t):
        if self.Rshock == None:
            return calc_shock_radius(t=t, E_SN=self.E_SN, n0=self.n0)
        else:
            return self.Rshock
    

class HadronicImpulsiveTimePDF(BaseHadronicImpulsivePDF):

    def __init__(self, n=100/u.cm**3, n0=0.3/u.cm**3,
                 D0=3e27*u.cm**2/u.s, dif_chi=0.01, dif_delta=0.5, deltap=0.3, 
                 Mej=1.4*u.solMass, Emax_t_ST0=1*u.PeV, 
                 E_SN=1e51*u.erg):
        
        super(HadronicImpulsiveTimePDF, self).__init__(n, n0,
                                                       D0, dif_chi, dif_delta, deltap,
                                                       Mej, Emax_t_ST0, E_SN)
        
        self.name = 'AharonianImpulsiveTime'
        
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def dif_time(self, E, t):
        tesc = calc_escape_time(E, deltap=self.deltap, t_ST0=self.t_ST0, Emax_t_ST0=self.Emax_t_ST0)
        t = t * np.ones(np.shape(tesc))
        return t - tesc
    
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def Rsh(self, E, t):
        tesc = calc_escape_time(E, deltap=self.deltap, t_ST0=self.t_ST0, Emax_t_ST0=self.Emax_t_ST0)
        Rsh = calc_shock_radius(t=tesc, E_SN=self.E_SN, n0=self.n0)
        return Rsh
    
    @u.quantity_input(t=u.kyr)
    def Rcrit(self, t):
        Rcrit = calc_shock_radius(t=t, E_SN=self.E_SN, n0=self.n0)
        return Rcrit


class HadronicImpulsiveTimeExtentPDF(BaseHadronicImpulsivePDF):

    def __init__(self, n=100/u.cm**3, n0=0.3/u.cm**3,
                 D0=3e27*u.cm**2/u.s, dif_chi=0.01, dif_delta=0.5, deltap=0.3, 
	             Mej=1.4*u.solMass, Emax_t_ST0=1*u.PeV, 
                 E_SN=1e51*u.erg):
        
        super(HadronicImpulsiveTimeExtentPDF, self).__init__(n, n0,
            D0, dif_chi, dif_delta, deltap, 
            Emax_obs, 
            t_ST, Mej, Emax_t_ST0, E_SN)
        
        self.name = 'AharonianImpulsiveTimeExtent'
        
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def dif_l(self, E, t):
        self.B = calc_Bmag(self.n)
        dif_coeff = self.dif_chi * self.D0 * (E / u.GeV / (self.B / (3*u.uG))) ** self.dif_delta
        dif_l = 2 * np.sqrt(dif_coeff * self.dif_time(E,t))
        #shock_extent = 0.909 * self.Resc # analytical solution
        shock_extent = 0.0875 * self.Resc # theoretical assumption
        return dif_l + shock_extent
        
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def dif_time(self, E, t):
        tesc = calc_escape_time(E, deltap=self.deltap, t_ST0=self.t_ST0, Emax_t_ST0=self.Emax_t_ST0)
        t = t * np.ones(np.shape(tesc))
        return t - tesc
    
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def Rsh(self, E, t):
        tesc = calc_escape_time(E, deltap=self.deltap, t_ST0=self.t_ST0, Emax_t_ST0=self.Emax_t_ST0)
        Rsh = calc_shock_radius(t=tesc, E_SN=self.E_SN, n0=self.n0)
        return Rsh
    
    @u.quantity_input(t=u.kyr)
    def Rcrit(self, t):
        Rcrit = calc_shock_radius(t=t, E_SN=self.E_SN, n0=self.n0)
        return Rcrit



