import numpy as np
from astropy import units as u
from astropy.constants import c, m_p
from astropy.io import fits

mp = (m_p * c**2).to('MeV')
mpi0 = 134.976*u.MeV

class GammaFlux:

    @u.quantity_input(nH2_cell=u.cm ** (-3))
    def __init__(self, acc, pdf_cube_path, gp, alpha=2, eta=0.1, xsection='Kelner', depth=35*u.pc):
        """
        A class to calculate the predicted gamma-ray flux

        """
        self.acc = acc
        
        self.pdf_cube_path = pdf_cube_path
        file = fits.open(self.pdf_cube_path)
        self.header = file[0].header
        self.pdf = (file[0].data) * u.Unit(self.header['BUNIT'])
        
        self.alpha = alpha
        self.eta = eta
        self.Ebudget = self.eta * self.header['HESN_ERG']*u.erg
        
        self.xsection = xsection
        
        self.depth = depth
        
        Emin = 1 * u.GeV
        Emax  = self.header['HEMT0_P']*u.PeV
        if self.alpha == 2:
            N0 = (self.Ebudget / ((np.log(Emax/u.GeV) - np.log(Emin/u.GeV))*u.GeV)).decompose()
        else:
            N0 = (self.Ebudget * (2-alpha) / 
                  (((Emax/u.GeV)**(2-self.alpha) - (Emin/u.GeV)**(2-self.alpha))*u.GeV)).decompose()
        self.N0 = N0
        
        Ep = np.logspace(self.header['CRVAL3'],
                         self.header['CRVAL3'] + (self.header['NAXIS3'] - 1) * self.header['CDELT3'],
                         self.header['NAXIS3'])* u.Unit(self.header['CUNIT3'])
        self.Ep = Ep + mp
                              
        if self.xsection=='Kelner':
            L = np.log(self.Ep / u.TeV)
            s_inel = ((34.3 + 1.88 * L + 0.25 * L ** 2) * u.mbarn).to(u.cm ** 2)
        elif self.xsection=='Kafexhiu':
            Ep_th = (2 * mpi0 + mpi0**2 / (2*mp)).to(u.GeV)
            L = np.log(self.Ep / Ep_th)
            s_inel = ((30.7 - 0.96 * L + 0.18 * L**2 ) * (1 - (Ep_th / self.Ep)**(1.9))**3  * u.mbarn).to(u.cm ** 2)

        kappa = 0.45
        cool_time = ((self.header['HN_CM3'] * u.cm**(-3) * c * kappa * s_inel)**(-1)).to(u.kyr)
        losses = np.exp(- ((self.alpha - 1) * self.header['HT_KYR']*u.kyr  / cool_time)) 

        Esize, xsize, ysize = self.pdf.shape
        Jp = np.empty((Esize, xsize, ysize))
        for i in range(len(self.Ep)):
            Jp[i, :,:] = (self.pdf[i,:,:] * losses[i] * N0 * (self.Ep[i] / u.GeV)**(-self.alpha) /u.GeV).to(u.cm**(-3)/u.GeV)

        self.Jp = Jp *  u.cm**(-3)/u.GeV
            
        Resc = self.header['HRESC_PC']*u.pc
        n0 = self.header['HN0_CM3'] * u.cm**(-3)
        self.gp = gp
        Rs = gp.dist_cell()
        self.nH2_cell = gp.nH2_cell(depth=self.depth).to(u.cm**(-3)).value
        self.nH2_cell[Rs < Resc] = n0
        self.nH2_cell = self.nH2_cell * u.cm**(-3)      
        
        if type(self.Ep.value) == np.ndarray:
            self.Ep = self.Ep[:, None, None]


    def calc_phi(self, Egamma):

        index = (np.abs(self.Ep - Egamma)).argmin()
        Jp = self.Jp[index:]
        Ep = self.Ep[index:]
        
        # Calculation of cross section of pp collisions
        if self.xsection=='Kelner':
            L = np.log(Ep / u.TeV)
            s_inel = ((34.3 + 1.88 * L + 0.25 * L ** 2) * u.mbarn).to(u.cm ** 2)
        elif self.xsection=='Kafexhiu':
            Ep = Ep + mp
            Ep_th = (2 * mpi0 + mpi0**2 / (2*mp)).to(u.GeV)
            L = np.log(Ep / Ep_th)
            s_inel = ((30.7 - 0.96 * L + 0.18 * L**2 ) * (1 - (Ep_th / Ep)**(1.9))**3  * u.mbarn).to(u.cm ** 2)

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

        # Converting the pixel width from angular size to physical size
        pixel_width = self.acc.distance.to(u.cm) * np.tan(self.gp.theta_pixel)
        ster = (pixel_width ** 2 / (self.acc.distance.to(u.cm)) ** 2) * u.sr

        Gfluxes = []

        for Eg in energies:
            Gflux = self.calc_phi(Eg) * self.depth * ster / (4 * np.pi * u.sr)
            unit = Gflux.unit
            Gfluxes += [Gflux]

        self.energies = energies
        self.fluxes = (Gfluxes * unit).to(1 / (u.TeV * u.s * u.cm**2))

        return

    def save_header(self, author='', extra_comments=None):
        """
        Save the hadronic gamma ray header
        """

        step = np.diff(np.log10(self.energies[:].to(u.TeV).value))[0]

        E0 = np.log10(self.energies[0].to(u.TeV).value)

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

        hdr.header['HNAME'] = self.header['HNAME']
        hdr.header['HT_KYR'] = self.header['HT_KYR']
        hdr.header['HCHI'] = self.header['HCHI']
        hdr.header['HDELTA'] = self.header['HDELTA']
        hdr.header['HD0_CM2S'] = self.header['HD0_CM2S']
        hdr.header['HN_CM3'] = self.header['HN_CM3']
        hdr.header['HN0_CM3'] = self.header['HN0_CM3']
        hdr.header['HESN_ERG'] = self.header['HESN_ERG']
        hdr.header['HEMT_TEV'] = self.header['HEMT_TEV']
        hdr.header['HEMT0_P'] = self.header['HEMT0_P']
        hdr.header['HTST0_YR'] = self.header['HTST0_YR']
        hdr.header['HDELTAP'] = self.header['HDELTAP']
        hdr.header['HEB_ERG'] = self.Ebudget.to(u.erg).value
        hdr.header['HETA'] = self.eta
        hdr.header['HALPHA'] = self.alpha
        hdr.header['HXSEC'] = self.xsection
        
        hdr.header['HMEJ_SOL'] = self.header['HMEJ_SOL']
        hdr.header['HSN_TYPE'] = self.header['HSN_TYPE']

        hdr.header['HEES_TEV'] = self.header['HEES_TEV']
        hdr.header['HRESC_PC'] = self.header['HRESC_PC']

        hdr.header['HMEJ_SOL'] = self.header['HMEJ_SOL']
        hdr.header['HSN_TYPE'] = self.header['HSN_TYPE']

        hdr.header['HEES_TEV'] = self.header['HEES_TEV']
        hdr.header['HRESC_PC'] = self.header['HRESC_PC']

        hdr.header['HGASPATH'] = self.header['HGASPATH']

        if extra_comments != None:
            for key in extra_comments.keys():
                hdr.header[key] = extra_comments[key]

        hdr.header['AUTHOR'] = author

        return hdr.header
    
    def save(self, author='', basename='HadronicFlux', extra_comments=None):
        """
        Save the hadronic gamma ray flux to a fits file
        """
        try:
            basepdf = self.pdf_cube_path.split('HadronicPDF_')[1].split('.fits')[0]
            filename = f'{basename}_{basepdf}_alpha{self.alpha}.fits'

            step = np.diff(np.log10(self.energies[:].to(u.TeV).value))[0]

            E0 = np.log10(self.energies[0].to(u.TeV).value)

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

            hdr.header['HNAME'] = self.header['HNAME']
            hdr.header['HT_KYR'] = self.header['HT_KYR']
            hdr.header['HCHI'] = self.header['HCHI']
            hdr.header['HDELTA'] = self.header['HDELTA']
            hdr.header['HD0_CM2S'] = self.header['HD0_CM2S']
            hdr.header['HN_CM3'] = self.header['HN_CM3']
            hdr.header['HN0_CM3'] = self.header['HN0_CM3']
            hdr.header['HESN_ERG'] = self.header['HESN_ERG']
            hdr.header['HEMT_TEV'] = self.header['HEMT_TEV']
            hdr.header['HEMT0_P'] = self.header['HEMT0_P']
            hdr.header['HTST0_YR'] = self.header['HTST0_YR']
            hdr.header['HDELTAP'] = self.header['HDELTAP']
            hdr.header['HEB_ERG'] = self.Ebudget.to(u.erg).value
            hdr.header['HETA'] = self.eta
            hdr.header['HALPHA'] = self.alpha
            hdr.header['HXSEC'] = self.xsection
            
            hdr.header['HMEJ_SOL'] = self.header['HMEJ_SOL']
            hdr.header['HSN_TYPE'] = self.header['HSN_TYPE']

            hdr.header['HEES_TEV'] = self.header['HEES_TEV']
            hdr.header['HRESC_PC'] = self.header['HRESC_PC']

            hdr.header['HGASPATH'] = self.header['HGASPATH']

            if extra_comments != None:
                for key in extra_comments.keys():
                    hdr.header[key] = extra_comments[key]

            hdr.header['AUTHOR'] = author

            hdr.writeto(filename, overwrite=True)
        except:
            print('Please calculate the flux first')
