import numpy as np
from astropy import units as u
from astropy.constants import c, m_p
from astropy.io import fits

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.integrate import simps
from .spatialmodel import SpatialModel3D, SpatialModel, oversample

mp = (m_p * c**2).to('MeV')
mpi0 = 134.976*u.MeV

class GammaFlux:

    def __init__(self, acc, pdf_cube_path, gp, alpha=2, eta=0.1, xsection='Kelner', factor=1): 
        """
        A class to calculate the predicted gamma-ray flux

        """
        self.acc = acc
        
        self.pdf_cube_path = pdf_cube_path
        file = fits.open(self.pdf_cube_path)
        self.header = file[0].header
        self.pdf = (file[0].data) * u.Unit(self.header['BUNIT'])
        
        Resc = self.header['HRESC_PC']*u.pc
        n0 = self.header['HN0_CM3'] * u.cm**(-3)
        
        self.alpha = alpha
        self.eta = eta
        self.Ebudget = self.eta * self.header['HESN_ERG']*u.erg
        
        self.xsection = xsection
        
        self.depth = 2 * Resc
        
        Emin = 1 * u.GeV
        Emax = self.header['HEMAX_P']*u.PeV
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

        Esize, xsize, ysize = self.pdf.shape
        Jp = np.empty((Esize, xsize, ysize))
        for i in range(len(self.Ep)):
            Jp[i, :,:] = (self.pdf[i,:,:] * N0 * (self.Ep[i] / u.GeV)**(-self.alpha) /u.GeV).to(u.cm**(-3)/u.GeV)

        self.Jp = Jp *  u.cm**(-3)/u.GeV
            
        self.gp = gp
        Rs = gp.dist_cell()
        
        self.NH_cell = gp.total_col_dens.to(u.cm**(-2))
        self.NH_cell = np.float64(self.NH_cell)
        self.NH_cell[self.NH_cell < 0] = 0
        
        # Include potential factor in NH_cell
        self.NH_cell[Rs < Resc] = (factor * self.NH_cell)[Rs < Resc] 

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
            Ep_th = (2 * mpi0 + mpi0**2 / (2*mp)).to(u.GeV)
            L = np.log(Ep / Ep_th)
            s_inel = ((30.7 - 0.96 * L + 0.18 * L**2 ) * (1 - (Ep_th / Ep)**(1.9))**3  * u.mbarn).to(u.cm ** 2)

        # Parameters for gamma ray spectrum
        B_g = 1.3 + 0.14 * L + 0.011 * L ** 2
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
        
        phi_g = (c * s_inel * Jp * Fg / Ep)
        phi_g.to(1 / (u.TeV ** 2 * u.s))
        
        phi = np.sum(phi_g[:-1] * np.diff(Ep, axis=0), axis=0).to(1 / (u.TeV * u.s))
        
#        print(Ep)


        return phi

    def flux(self, energies):

        # Converting the pixel width from angular size to physical size
        pixel_width = self.acc.distance.to(u.cm) * np.tan(self.gp.theta_pixel)
        A = pixel_width ** 2
        pixel_ster = (A / self.acc.distance ** 2).decompose() * u.sr

        Gfluxes = []

        for Eg in energies:
            Gflux = self.calc_phi(Eg) * self.NH_cell * pixel_ster /  (4 * np.pi * u.sr)
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
        hdr.header['HEMAX_P'] = self.header['HEMAX_P']
        hdr.header['HTSED_YR'] = self.header['HTSED_YR']
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
            filename = f'{basename}_{basepdf}_alpha{self.alpha}_eta{self.eta:.3f}.fits'

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
            hdr.header['HEMAX_P'] = self.header['HEMAX_P']
            hdr.header['HTSED_YR'] = self.header['HTSED_YR']
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
            
            
            

class HadronicGammaMaker:

    def __init__(self, pdf_cube_path, acc, gp, alpha=2, eta=1):

        """
        Parameters
        ----------------------
        pdf_cube_path : path
            The path for the hadronic impulsive PDF
        acc : class
            The class for the accelerator
        gp : class
            The class for the gas parameters
        eta : float
            The scaling value for the energy budget
        alpha : float
            The spectra index

        """
        
        self.pdf_cube_path = pdf_cube_path
        self.acc = acc
        self.gp = gp
        self.alpha = alpha
        self.eta = eta
        
            
        gf = GammaFlux(acc=self.acc, pdf_cube_path=self.pdf_cube_path, gp=self.gp, 
                           alpha=self.alpha, eta=self.eta)
        self.gf = gf   

    def plot_gamma_flux(self, Egamma, cmap='viridis'):

        Eg=[Egamma.value]*Egamma.unit
        self.gf.flux(Eg)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection=WCS(self.gp.header))
        img = ax.imshow(self.gf.fluxes[0,:,:].value, cmap=cmap)
        cb = fig.colorbar(img, pad=0.02)
        cb.set_label(label=f'Gamma flux @ {self.gf.energies[0].value:.0f} TeV ({self.gf.fluxes.unit})', 
                     rotation=270, labelpad=15)
        lon = ax.coords['glon']
        lat = ax.coords['glat']
        lon.set_axislabel('Galactic Longitude')
        lat.set_axislabel('Galactic Latitude')
        lat.set_major_formatter('d.d')
        lon.set_major_formatter('d.d')
        lon.set_ticklabel_position('b')

        return ax
    
    
    def calc_integrated_flux(self, Egmin, Egmax):

        Egammas = np.logspace(np.log10((Egmin.to(u.TeV)).value), 
                             np.log10((Egmax.to(u.TeV)).value), 100)*u.TeV

        self.gf.flux(Egammas)
        fluxes = self.gf.fluxes.value
        header = self.gf.save_header()
        sm3D = SpatialModel3D(fluxes, header, energies=Egammas)

        Esize, xsize, ysize = fluxes.shape

        energies = sm3D.energies

        integrated = np.empty((xsize, ysize))
        for ix in range(xsize):
            for iy in range(ysize):
                if np.isfinite(fluxes[:, ix, iy]).all():
                    integrated[ix, iy] = simps(fluxes[:, ix, iy], energies)
                else:
                    integrated[ix, iy] = 0.0

        return integrated, header
    
    
    def plot_gamma_integral_flux(self, Egmin, Egmax, cmap='viridis'):

        integrated, header = self.calc_integrated_flux(Egmin, Egmax)

        hdu = fits.PrimaryHDU(data=integrated)
        hdu.header = header
        hdu.header.remove('CTYPE3')
        hdu.header.remove('CUNIT3')
        hdu.header.remove('CRVAL3')
        hdu.header.remove('CRPIX3')
        hdu.header.remove('CDELT3')

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection=WCS(header))
        img = ax.imshow(integrated, cmap=cmap)
        cb = fig.colorbar(img, pad=0.02)
        cb.set_label(label=f'Integral Gamma-ray Flux ({Egmin.to(u.TeV).value:.0f}-{Egmax.to(u.TeV).value:.0f} TeV)'
                     +' cm$^{-2}$ s$^{-1}$', 
                     rotation=270, labelpad=15)
        lon = ax.coords['glon']
        lat = ax.coords['glat']
        lon.set_axislabel('Galactic Longitude')
        lat.set_axislabel('Galactic Latitude')
        lat.set_major_formatter('d.d')
        lon.set_major_formatter('d.d')
        lon.set_ticklabel_position('b')

        return ax
    
    def plot_oversample_gamma_integral_flux(self, Egmin, Egmax, 
                                            grid_size=0.02*u.deg, radius=0.1*u.deg, cmap='viridis'):

        integrated, header = self.calc_integrated_flux(Egmin, Egmax)

        hdu = fits.PrimaryHDU(data=integrated)
        hdu.header = header
        hdu.header.remove('CTYPE3')
        hdu.header.remove('CUNIT3')
        hdu.header.remove('CRVAL3')
        hdu.header.remove('CRPIX3')
        hdu.header.remove('CDELT3')

        oversampled_header, oversampled_data = oversample(integrated, hdu.header, 
                                                          grid_size=grid_size, radius=radius)
        self.oversampled_header = oversampled_header
        
        smI = SpatialModel(data=oversampled_data, header=oversampled_header)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection=WCS(oversampled_header))
        img = ax.imshow(oversampled_data, cmap=cmap)
        cb = fig.colorbar(img, pad=0.02)
        cb.set_label(label=f'Integral Gamma-ray Flux ({Egmin.to(u.TeV).value:.0f} - {Egmax.to(u.TeV).value:.0f} TeV)'
                     +' cm$^{-2}$ s$^{-1}$', 
                     rotation=270, labelpad=15)
        lon = ax.coords['glon']
        lat = ax.coords['glat']
        lon.set_axislabel('Galactic Longitude')
        lat.set_axislabel('Galactic Latitude')
        lat.set_major_formatter('d.d')
        lon.set_major_formatter('d.d')
        lon.set_ticklabel_position('b')

        return ax
    
    def plot_gamma_spectra(self, sed_type, Egmin, Egmax, 
                           glon, glat, radius, frame, 
                           fig=None, ax=None, **kwargs): 
        
        Egammas = np.logspace(np.log10((Egmin.to(u.TeV)).value), 
                              np.log10((Egmax.to(u.TeV)).value), 100)*u.TeV
        self.gf.flux(Egammas)
        fluxes = self.gf.fluxes.value
        header = self.gf.save_header()
        sm3D = SpatialModel3D(fluxes, header, energies=Egammas)
        energy, flux = sm3D.sum_region(coords=[glon, glat], radius=radius)

        OBJ_coord = SkyCoord(glon, glat, frame=frame)

        file = fits.open(self.gp.gas_map_path)
        total_col_dens = file[0].data
        header_gas = file[0].header
        xvalue, yvalue = total_col_dens.shape
        xx, yy = np.meshgrid(range(yvalue), range(xvalue))
        wx, wy = WCS(header_gas).wcs_pix2world(xx, yy, 0)
        pixel_coord = SkyCoord(wx * u.deg, wy * u.deg, frame='galactic')

        theta_sep_hess = (OBJ_coord.separation(pixel_coord)).deg * u.deg
        R_hess_vals = self.acc.distance.to(u.cm) * np.tan(theta_sep_hess)

        d_hess = self.acc.distance * np.tan(radius)
        R_hess_less = theta_sep_hess < radius

        R_vals = self.gp.dist_cell()
        R_esc_less = R_vals < header['HRESC_PC']*u.pc

        # Bubble component
        flux_bubble = np.empty((fluxes.shape[0]))
        for ii in range(fluxes.shape[0]):
            flux_bubble[ii] = np.nansum(fluxes[ii, R_hess_less & R_esc_less])

        # Diffused component
        flux_diffused = np.empty((fluxes.shape[0]))
        for kk in range(fluxes.shape[0]):
            flux_diffused[kk] = np.nansum(fluxes[kk, R_hess_less & ~R_esc_less])


        if fig is None:
            fig = plt.figure(figsize=(6, 4))

        if ax is None:
            ax = fig.add_subplot(111)


        if sed_type == 'dnde':
            ax.plot(energy, flux, label='Total emission', color='blue', ls='-')
            ax.plot(energy, flux_bubble, label='Bubble emission', color='red', ls='-.')
            ax.plot(energy, flux_diffused, label='Diffused emission', color='black', ls='--')
            ax.set_xlim([(Egmin.to(u.TeV)).value, (Egmax.to(u.TeV)).value])
            ax.grid(ls='--')
            ax.set_xlabel('Energy (TeV)')
            ax.set_ylabel('Gamma-ray flux (TeV$^{-1}$cm$^{-2}$s$^{-1}$)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(loc='lower left', labelspacing=0.3)

        if sed_type == 'e2dnde':
            ax.plot(energy, energy**2*flux, label='Total emission', color='blue', ls='-')
            ax.plot(energy, energy**2*flux_bubble, label='Bubble emission', color='red', ls='-.')
            ax.plot(energy, energy**2*flux_diffused, label='Diffused emission', color='black', ls='--')
            ax.set_xlim([(Egmin.to(u.TeV)).value, (Egmax.to(u.TeV)).value])
            ax.grid(ls='--')
            ax.set_xlabel('Energy (TeV)')
            ax.set_ylabel('Gamma-ray energy flux (TeV cm$^{-2}$s$^{-1}$)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(loc='lower left', labelspacing=0.3)

        return ax


