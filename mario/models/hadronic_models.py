import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.constants import M_sun, m_p, c
from astropy.wcs import WCS
import matplotlib.pyplot as plt

from .utilities import BasePDF
from .utilities import calc_Bmag, calc_shock_radius, calc_energy_evolution
from .utilities import calc_escape_time, calc_escape_energy

class BaseHadronicImpulsive(BasePDF):
    """
    A base class that can be used to create hadronic PDFs
    """

    def __init__(self, n=100*u.cm**-3, n0=1*u.cm**-3,  
                 D0=3e27*u.cm**2/u.s, dif_chi=0.01, dif_delta=0.5, deltap=0.3, 
                 Mej=1.0*u.solMass, Epmin=1*u.GeV, Epmax=1*u.PeV, 
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
        Epmin : float, optional
            Minimum energy of cosmic ray protons at t (default is 1 GeV)
        Epmax : float, optional
            Maximum injection energy of cosmic ray protons at t_sedov (corresponds to start of 
            Sedov-Taylor phase) (default is 1 PeV)
        E_SN : float, optional
            Kinetic energy of SN explosion (default is 10^51 erg)

        """
        super(BaseHadronicImpulsive, self).__init__(E_SN)
        
        self.n = n
        self.n0 = n0
        self.D0 = D0
        self.dif_chi = dif_chi
        self.dif_delta = dif_delta
        
        self.E_SN = E_SN
        
        self.deltap = deltap
        self.rho = 1 * m_p * u.cm**(-3)
        self.Mej = Mej
        if self.Mej < 1.44*u.solMass:
        	self.SN_type = 'Type Ia'
        elif self.Mej > 8*u.solMass:
        	self.SN_type = 'Type II'
        self.t_sedov = 1.6e3*u.yr * (self.E_SN / (1e51*u.erg))**(-1/2) * (self.Mej / (10*u.solMass))**(5/6) * (self.rho / (m_p * u.cm**(-3)))**(-1/3)

        self.Epmin = Epmin        
        self.Epmax = Epmax
        
        
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
        
        Ecrit = self.Ecrit(t)
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
        
        self.Emax_t = (self.Epmax * ((t / self.t_sedov).decompose()) ** (-self.deltap)).to(u.TeV)
        self.R = R
        self.t = t
        self.energy = E[:,0,0]
        self.pdf = pdf
        
        
    def save(self, author='', basename='HadronicPDF', extra_comments=None):
        """
        Save the hadronic PDF to a fits file
        """
        try:
            filename = f'{basename}_{self.name}_chi{self.dif_chi}_delta{self.dif_delta}_D0{self.D0.to(u.cm**2/u.s).value}_n{self.n.to(u.cm**-3).value:.0f}_n0{float(self.n0.to(u.cm**-3).value):.3f}_ESN{self.E_SN.to(u.erg).value}_Mej{self.Mej.to(u.solMass).value}_Epmax{self.Epmax.to(u.PeV).value}t_sedov{self.t_sedov.to(u.yr).value:.2f}_deltap{self.deltap}.fits'

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
            hdr.header['HN0_CM3'] = float(self.n0.to(u.cm**-3).value)
            hdr.header['HESN_ERG'] = self.E_SN.to(u.erg).value
            hdr.header['HEMT_TEV'] = self.Emax_t.to(u.TeV).value
            hdr.header['HEMAX_P'] = self.Epmax.to(u.PeV).value
            hdr.header['HTSED_YR'] = self.t_sedov.to(u.yr).value
            hdr.header['HDELTAP'] = self.deltap
            
            hdr.header['HMEJ_SOL'] = self.Mej.to(u.solMass).value
            hdr.header['HSN_TYPE'] = self.SN_type
            
            hdr.header['HEES_TEV'] = self.Eesc.to(u.TeV).value
            hdr.header['HRESC_PC'] = self.Resc[0,0,0].to(u.pc).value

            hdr.header['AUTHOR'] = author
            
            if extra_comments != None:
                for key in extra_comments.keys():
                    hdr.header[key] = extra_comments[key]

            hdr.writeto(filename, overwrite=True)
        except:
            print('Please calculate the pdf first')
 

class AharonianImpulsive(BaseHadronicImpulsive):

    def __init__(self, n=100/u.cm**3, n0=1/u.cm**3,
                 D0=3e27*u.cm**2/u.s, dif_chi=0.01, dif_delta=0.5, deltap=0.3,
                 Mej=1.4*u.solMass, Epmin=1*u.GeV, Epmax=1*u.PeV, 
                 E_SN=1e51*u.erg,
                 Rshock=None, D10=1e26*u.cm**2/u.s):
        
        super(AharonianImpulsive, self).__init__(n, n0,
            D0, dif_chi, dif_delta, deltap,
            Mej, Epmin, Epmax, E_SN)
        
        self.Rshock = Rshock
        self.D10 = D10
        
        self.name = 'AharonianImpulsive'

    @u.quantity_input(E=u.GeV, t=u.kyr)
    def dif_l(self, E, t):
        dif_coeff = self.D10 * (E / (10*u.GeV)) ** self.dif_delta
        dif_l = 2 * np.sqrt(dif_coeff * self.dif_time(E,t))
        return dif_l

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

    @u.quantity_input(t=u.kyr)
    def Ecrit(self, t):
        Ecrit = 0*u.TeV
        return Ecrit       


class HadronicImpulsive(BaseHadronicImpulsive):

    def __init__(self, n=100/u.cm**3, n0=1/u.cm**3,
                 D0=3e27*u.cm**2/u.s, dif_chi=0.01, dif_delta=0.5, deltap=0.3,
                 Mej=1.4*u.solMass, Epmin=1*u.GeV, Epmax=1*u.PeV,
                 E_SN=1e51*u.erg,
                 Rshock=None):
        
        super(HadronicImpulsive, self).__init__(n, n0,
            D0, dif_chi, dif_delta, deltap,
            Mej, Epmin, Epmax, E_SN)
        
        self.Rshock = Rshock
        
        self.name = 'HadronicImpulsive'
        
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

    @u.quantity_input(t=u.kyr)
    def Ecrit(self, t):
        Ecrit = 0*u.TeV
        return Ecrit       
    

class HadronicImpulsiveTime(BaseHadronicImpulsive):

    def __init__(self, n=100/u.cm**3, n0=0.3/u.cm**3,
                 D0=3e27*u.cm**2/u.s, dif_chi=0.01, dif_delta=0.5, deltap=0.3, 
                 Mej=1.4*u.solMass, Epmin=1*u.GeV, Epmax=1*u.PeV,
                 E_SN=1e51*u.erg):
        
        super(HadronicImpulsiveTime, self).__init__(n, n0,
                                                       D0, dif_chi, dif_delta, deltap,
                                                       Mej, Epmin, Epmax, E_SN)
        
        self.name = 'HadronicImpulsiveTime'
        
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def dif_l(self, E, t, extent=False):
        self.B = calc_Bmag(self.n)
        dif_coeff = self.dif_chi * self.D0 * (E / u.GeV / (self.B / (3*u.uG))) ** self.dif_delta
        dif_l = 2 * np.sqrt(dif_coeff * self.dif_time(E,t))
        if extent==True:
            #shock_extent = 0.909 * self.Resc # analytical solution
            shock_extent = 0.0875 * self.Resc # theoretical assumption
        else:
            shock_extent = 0
        return dif_l + shock_extent
        
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def dif_time(self, E, t):
        tesc = calc_escape_time(E, deltap=self.deltap, t_sedov=self.t_sedov, Epmax=self.Epmax)
        t = t * np.ones(np.shape(tesc))
        return t - tesc
    
    @u.quantity_input(E=u.GeV, t=u.kyr)
    def Rsh(self, E, t):
        tesc = calc_escape_time(E, deltap=self.deltap, t_sedov=self.t_sedov, Epmax=self.Epmax)
        Rsh = calc_shock_radius(t=tesc, E_SN=self.E_SN, n0=self.n0)
        return Rsh
    
    @u.quantity_input(t=u.kyr)
    def Rcrit(self, t):
        Rcrit = calc_shock_radius(t=t, E_SN=self.E_SN, n0=self.n0)
        return Rcrit

    @u.quantity_input(t=u.kyr)
    def Ecrit(self, t):
        Ecrit = calc_escape_energy(t, deltap=self.deltap, t_sedov=self.t_sedov, Epmax=self.Epmax) 
        return Ecrit


class HadronicImpulsiveMaker:

    def __init__(self, ps, acc, gp, eta, alpha):

        """
        Parameters
        ----------------------
        ps : class
            The class for the hadronic impulsive PDF
        acc : class
            The class for the accelerator
        gp : class
            The class for the gas parameters
        eta : float
            The scaling value for the energy budget
        alpha : float
            The spectra index

        """
        
        self.ps = ps
        self.acc = acc
        self.gp = gp
        
        self.eta = eta
        self.alpha = alpha
        
        self.Ebudget = self.eta * self.ps.E_SN    
        
        if alpha==2:
            N0 = (self.Ebudget / ((np.log(self.ps.Epmax/u.GeV) - np.log(self.ps.Epmin/u.GeV))*u.GeV)).decompose()
        else:
            N0 = (self.Ebudget * (2-alpha) / 
                  (((self.ps.Epmax/u.GeV)**(2-alpha) - (self.ps.Epmin/u.GeV)**(2-alpha))*u.GeV)).decompose()
        
        self.N0 = N0

    def calc_proton_spectra(self, energies, radius):

        self.ps.calc_pdf(E=energies, R=radius, t=self.acc.age)

        cool_time = (6e7 * (self.ps.n/ u.cm**-3) ** (-1) * u.yr)
        losses = np.exp(- ((self.alpha - 1) * self.ps.t  / cool_time))

        J = (self.ps.pdf[:,0,0] * losses * self.N0 * (self.ps.energy / u.GeV)**(-self.alpha) /u.GeV * 
            self.ps.energy** 2.75 * c / (4 * np.pi * u.sr)).to(u.GeV ** (7 / 4) / (u.cm ** 2 * u.s * u.sr))

        return J
    
    def plot_proton_spectra(self, energies, radius):
        
        J = self.calc_proton_spectra(energies=energies, radius=radius)
        
        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()

        ax.plot(energies, J)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('Energy (GeV)')
        ax.set_ylabel('JE$^{2.75}$ (GeV$^{1.75}$cm$^{-2}$s$^{-1}$ sr$^{-1}$)')

        return ax
    
    def calc_proton_flux(self, energy):
    
        Ep = [energy.value]*energy.unit
        self.ps.calc_pdf(E=Ep, R=self.gp.dist_cell(), t=self.acc.age)

        Jp = (self.ps.pdf[0,:,:] * self.N0 * (self.ps.energy[0] / u.GeV)**(-self.alpha) /u.GeV * c / (4 * np.pi * u.sr)).to(1 / (u.TeV * u.cm ** 2 * u.s * u.sr))
        return Jp
    
    def plot_proton_flux(self, energy, cmap='Greys'):
    
        Jps = self.calc_proton_flux(energy=energy)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection=WCS(self.gp.header))
        img = ax.imshow(Jps.value, cmap=cmap)
        cb = fig.colorbar(img, pad=0.02)
        cb.set_label(label=f'Proton flux @ {energy.value:.0f} TeV ({Jps.unit})', rotation=270, labelpad=15)
        lon = ax.coords['glon']
        lat = ax.coords['glat']
        lon.set_axislabel('Galactic Longitude')
        lat.set_axislabel('Galactic Latitude')
        lat.set_major_formatter('d.d')
        lon.set_major_formatter('d.d')
        lon.set_ticklabel_position('b')

        return ax
