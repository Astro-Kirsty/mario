import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.io import fits

class GammaFlux:

    @u.quantity_input(nH2_cell=u.cm ** (-3))
    def __init__(self, snr, pdf_cube_path, gp, alpha=2, eta=0.1):
        """
        A class to calculate the predicted gamma-ray flux

        """
        self.snr = snr
        
        self.pdf_cube_path = pdf_cube_path
        file = fits.open(self.pdf_cube_path)
        self.header = file[0].header
        self.pdf = (file[0].data) * u.Unit(self.header['BUNIT'])
        
        self.alpha = alpha
        self.eta = eta
        self.Ebudget = self.eta * self.header['HESN_ERG']*u.erg
        
        Emin = 1 * u.GeV
        Emax = 1 * u.PeV
        if self.alpha == 2:
            N0 = (self.Ebudget / ((np.log(Emax/u.GeV) - np.log(Emin/u.GeV))*u.GeV)).decompose()
        else:
            N0 = (self.Ebudget * (2-alpha) / 
                  (((Emax/u.GeV)**(2-self.alpha) - (Emin/u.GeV)**(2-self.alpha))*u.GeV)).decompose()
        self.N0 = N0
        
        self.Ep = np.logspace(self.header['CRVAL3'],
                              self.header['CRVAL3'] + (self.header['NAXIS3'] - 1) * self.header['CDELT3'],
                              self.header['NAXIS3'])* u.Unit(self.header['CUNIT3'])
        
        cool_time = (6e7 * (self.header['HN_CM3'] * u.cm**-3 / u.cm**-3) ** (-1) * u.yr)
        losses = np.exp(- ((self.alpha - 1) * self.header['HT_KYR']*u.kyr  / cool_time)) 

        Esize, xsize, ysize = self.pdf.shape
        Jp = np.empty((Esize, xsize, ysize))
        for i in range(len(self.Ep)):
            Jp[i, :,:] = (self.pdf[i,:,:] * losses * N0 * (self.Ep[i] / u.GeV)**(-self.alpha) /u.GeV).to(u.cm**(-3)/u.GeV)

        self.Jp = Jp *  u.cm**(-3)/u.GeV
            
        Resc = self.header['HRESC_PC']*u.pc
        n0 = self.header['HN0_CM3'] * u.cm**(-3)
        self.gp = gp
        Rs = gp.dist_cell()
        self.nH2_cell = gp.nH2_cell().to(u.cm**(-3)).value
        self.nH2_cell[Rs < Resc] = n0
        self.nH2_cell = self.nH2_cell * u.cm**(-3)      
        
        if type(self.Ep.value) == np.ndarray:
            self.Ep = self.Ep[:, None, None]


    def calc_phi(self, Egamma):

        index = (np.abs(self.Ep - Egamma)).argmin()
        Jp = self.Jp[index:]
        Ep = self.Ep[index:]
        
        # Calculation of cross section of pp collisions
        L = np.log(Ep / u.TeV)
        s_inel = ((34.3 + 1.88 * L + 0.25 * L ** 2) * u.mbarn).to(u.cm ** 2)

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

        # Calculation of the fixed depth of some box surrounding the source
        dl = self.snr.distance.to(u.cm) * np.tan(0.45 * u.deg)

        # Converting the pixel width from angular size to physical size
        pixel_width = self.snr.distance.to(u.cm) * np.tan(self.gp.theta_pixel)
        ster = (pixel_width ** 2 / (self.snr.distance.to(u.cm)) ** 2) * u.sr

        Gfluxes = []

        for Eg in energies:
            Gflux = self.calc_phi(Eg) * dl * ster / (4 * np.pi * u.sr)
            unit = Gflux.unit
            Gfluxes += [Gflux]

        self.energies = energies
        self.fluxes = Gfluxes * unit

        return

    def save_header(self, author=''):
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

        hdr.header['HEES_TEV'] = self.header['HEES_TEV']
        hdr.header['HRESC_PC'] = self.header['HRESC_PC']

        hdr.header['HGASPATH'] = self.gp.gas_map_path

        hdr.header['AUTHOR'] = author

        return hdr.header
    
    def save(self, author='', basename='HadronicFlux'):
        """
        Save the hadronic gamma ray flux to a fits file
        """
        try:
            basepdf = self.pdf_cube_path.split('HadronicPDF_')[1]
            filename = f'{basename}_{basepdf}'

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

            hdr.header['HEES_TEV'] = self.header['HEES_TEV']
            hdr.header['HRESC_PC'] = self.header['HRESC_PC']

            hdr.header['HGASPATH'] = self.gp.gas_map_path

            hdr.header['AUTHOR'] = author

            hdr.writeto(filename, overwrite=True)
        except:
            print('Please calculate the flux first')
