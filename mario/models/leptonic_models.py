import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.constants import c, M_sun, m_p

from mario.models.utilities import BasePDF
from mario.models.utilities import calc_Bmag, calc_shock_radius, calc_energy_evolution
from mario.models.utilities import calc_escape_time, calc_escape_energy

class BaseLeptonicImpulsivePDF(BasePDF):
    """
    A base class that can be used to create leptonic PDFs
    """

    def __init__(self, n0=1*u.cm**-3,  
                 D10=1e28*u.cm**2/u.s, dif_delta=0.6, deltap=0.3,
                 Mej=1.4*u.solMass, Emax_t_ST0=1*u.PeV, 
                 E_SN=1e51*u.erg):

        super(BaseLeptonicImpulsivePDF, self).__init__(E_SN)
        
        self.D10 = D10
        self.dif_delta = dif_delta
        self.n0 = n0
        
        self.deltap = deltap
        self.rho = 1 * m_p * u.cm**(-3)
        self.Mej = Mej.to(u.kg)
        self.t_ST0 = 1.6e3*u.yr * (E_SN / (1e51*u.erg))**(-1/2) * (self.Mej / (10*M_sun))**(5/6) * (self.rho / (m_p * u.cm**(-3)))**(-1/3)
        
        self.Emax_t_ST0 = Emax_t_ST0
        
        # Electron mass
        me = (511 * u.keV).to(u.GeV) / c**2
        self.me = me
        
        # Synchrotron and Inverse compton losses
        wB = 0.5 * u.eV / u.cm**3
        wMBR = 0.25 * u.eV / u.cm**3
        wopt = 0.6 * u.eV / u.cm**3
        wo = wB + wMBR + wopt
        p2 = 5.2e-20 * (wo / (u.eV / u.cm**3) ) / u.s
        self.p2 = p2

        
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def dif_l(self, E, t):
        # Calculation of diffusion coefficient
        me = (511 * u.keV).to(u.GeV) / c**2
        gamma = E / (me * c ** 2)
        gamma_min = 3 * u.GeV / (me * c ** 2)
        gamma_max = 10 * u.GeV / (me * c ** 2)
        D0 = self.D10 / ((1 + (gamma_max /  gamma_min))**(self.dif_delta))
        dif_coeff = D0 * (1 + gamma / gamma_min)**(self.dif_delta)
        self.D0 = D0
        
        gamma_cut = (self.p2 * t)**(-1)
        frac = (1 - ((1 - gamma/gamma_cut)**(1 - self.dif_delta))) / ((1 - self.dif_delta) * gamma/gamma_cut)
        
        # Calculation of diffusion length
        dif_l = 2 * np.sqrt(dif_coeff * self.dif_time(E,t) * frac)
        
        self.gamma = gamma
        self.gamma_cut = gamma_cut
        
        return dif_l
    
    @u.quantity_input(E=u.GeV, R=u.pc)
    def calc_pdf(self, E, R, t):
        
        if type(E.value) == np.ndarray:
            E = E[:, None, None]
            if len(np.shape(R.value)) == 2:
                R = R * np.ones((len(E), np.shape(R)[0], np.shape(R)[1]))
            elif len(np.shape(R.value)) == 1:
                R = R * np.ones((len(E), np.shape(R)[0]))
                R = R[:, :, None]
        
        Rsh = self.Rsh(E, t) * np.ones(np.shape(R))

        Rcrit = self.Rcrit(t) * np.ones(np.shape(R))
        self.Resc = Rcrit
        
        # Define the critical radius and energy in which particles less than these value are trapped
        Ecrit = calc_escape_energy(t, deltap=self.deltap, t_ST0=self.t_ST0, Emax_t_ST0=self.Emax_t_ST0)
        self.Eesc = Ecrit
        
        # Calculation of diffusion length
        dif_length = self.dif_l(E,t)        
        pdf = np.zeros(np.shape(R)) / u.cm**3
        
        norm = ((np.sqrt(np.pi) * dif_length**2 + 2 * np.sqrt(np.pi) * Rsh**2) * np.abs(dif_length) + 4 * Rsh * dif_length**2) / (np.sqrt(np.pi) * dif_length**3)
        
        # Calculation of electron pdf 
        coeff = 1. / (np.pi ** (3 / 2) * dif_length ** 3) 
        pdf[(E >= Ecrit) & (R >= Rcrit)] = (coeff * np.exp(- (R - Rsh)**2 / dif_length**2 ) / norm)[(E >= Ecrit) & (R >= Rcrit)]
        
        # Particles with energies less than Ecrit have not escaped and are trapped
        pdf[(E < Ecrit) & (R < Rcrit)] = (1 / (4/3 * np.pi * Rcrit**3))[(E < Ecrit) & (R < Rcrit)]
            
        # Implement   γ > γ_cut  then  pdf = 0 
        gamma = self.gamma * np.ones(np.shape(pdf))
        gamma_cut = self.gamma_cut * np.ones(np.shape(pdf))
        pdf[(gamma > gamma_cut)] = 0*u.cm**(-3)
        
        self.Emax_t = (self.Emax_t_ST0 * ((t / self.t_ST0).decompose()) ** (-self.deltap)).to(u.TeV)
        self.R = R
        self.t = t
        self.energy = E[:,0,0]
        self.pdf = pdf    
        
    def save(self, author='', basename='LeptonicPDF'):
        """
        Save the leptonic PDF to a fits file
        """
        try:
            filename = f'{basename}_{self.name}_delta{self.dif_delta}_D0{self.D0.to(u.cm**2/u.s).value}_n0{self.n0.to(u.cm**-3).value}_ESN{self.E_SN.to(u.erg).value}_EmaxST0{self.Emax_t_ST0.to(u.PeV).value}_tST0{self.t_ST0.to(u.yr).value:.2f}_deltap{self.deltap}.fits'

            step = np.diff(np.log10(self.energy[:].decompose().value))[0]

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
            hdr.header['HDELTA'] = self.dif_delta
            hdr.header['HD0_CM2S'] = self.D0.to(u.cm**2/u.s).value
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
            
            
class LeptonicImpulsivePDF(BaseLeptonicImpulsivePDF):

    def __init__(self, n0=1/u.cm**3,
                 D10=1e28*u.cm**2/u.s, dif_delta=0.5, deltap=1.0,
                 Mej=1.4*u.solMass, Emax_t_ST0=1*u.PeV, 
                 E_SN=1e51*u.erg,
                 Rshock=None):
        
        super(LeptonicImpulsivePDF, self).__init__(n0, D10, dif_delta, deltap, Mej, Emax_t_ST0, E_SN)
        
        self.Rshock = Rshock
        
        self.name = 'AtoyanImpulsive'
        
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
            
class LeptonicImpulsiveTimePDF(BaseLeptonicImpulsivePDF):

    def __init__(self, n0=1*u.cm**-3,  
                 D10=1e28*u.cm**2/u.s, dif_delta=0.6, deltap=0.3,
                 Mej=1.4*u.solMass, Emax_t_ST0=1*u.PeV, 
                 E_SN=1e51*u.erg):
        
        super(LeptonicImpulsiveTimePDF, self).__init__(n0,
                                                       D10, dif_delta, deltap,
                                                       Mej, Emax_t_ST0, E_SN)
        
        self.name = 'AtoyanImpulsiveTime'
        
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
        
class LeptonicImpulsiveTimeExtentPDF(BaseLeptonicImpulsivePDF):

    def __init__(self, n0=1*u.cm**-3,  
                 D10=1e28*u.cm**2/u.s, dif_delta=0.6, deltap=0.3,
                 Mej=1.4*u.solMass, Emax_t_ST0=1*u.PeV, 
                 E_SN=1e51*u.erg):
        
        super(LeptonicImpulsiveTimeExtentPDF, self).__init__(n0,
            D10, dif_delta, deltap, 
            Mej, Emax_t_ST0, E_SN)
        
        self.name = 'AtoyanImpulsiveTimeExtent'
        
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def dif_l(self, E, t):
        # Calculation of diffusion coefficient
        me = (511 * u.keV).to(u.GeV) / c**2
        gamma = E / (me * c ** 2)
        gamma_min = 3 * u.GeV / (me * c ** 2)
        gamma_max = 10 * u.GeV / (me * c ** 2)
        D0 = self.D10 / ((1 + (gamma_max /  gamma_min))**(self.dif_delta))
        dif_coeff = D0 * (1 + gamma / gamma_min)**(self.dif_delta)
        self.D0 = D0
        gamma_cut = (self.p2 * t)**(-1)
        frac = (1 - ((1 - gamma/gamma_cut)**(1 - self.dif_delta))) / ((1 - self.dif_delta) * gamma/gamma_cut)
        
        # Calculation of diffusion length
        dif_l = 2 * np.sqrt(dif_coeff * self.dif_time(E,t) * frac)
        
        self.gamma = gamma
        self.gamma_cut = gamma_cut
        
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
