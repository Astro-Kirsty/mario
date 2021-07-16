"""
Example script to plot the gamma-ray fluxes in the hadronic scenario. To run:
python plot_gamma_fluxes.py 'HESSJ1804' 'G8_15' 'G8_Plot' HadronicFlux_AharonianImpulsiveTimeExtent.fits

This example produces figures for the hadronic scenario for gamma-rays. The spectral and spatial morphologies are plotted, along with the model and flux parameters, and a figure for the spatial residuals map. This example also saves a FluxParams.csv file which can be used to test which model is performing the best.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')

from scipy.integrate import simps

import astropy.units as u
from astropy import coordinates as coord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io.misc import yaml
from astropy.coordinates import SkyCoord
from photutils import SkyCircularAnnulus

import sys
import os
import click
import glob
import csv

from mario.sources import GammaSource, Accelerator
from mario.gamma.spatialmodel import SpectralModel, SpatialModel3D, SpatialModel
from mario.gamma.spatialmodel import oversample
from mario.plotting import add_SNR, add_SNR_centre, add_Resc, add_gamma_src, set_limits

@click.command()
@click.argument('gamma_source')
@click.argument('accelerator')
@click.argument('plot_type')
@click.argument('gamma_cubes')

@click.option('--obs_path',
               default='../data/hgps_map_flux_0.1deg_v1_crop.fits',
               type=click.Path(exists=True, dir_okay=False))

@click.option('--hess_spectra_path',
               default='../data/HESS_spectra_data.csv',
               type=click.Path(exists=True, dir_okay=False))
@click.option('--fermi_spectra_path',
               default='../data/Fermi_spectra_data.csv',
               type=click.Path(exists=True, dir_okay=False))

@click.option('--gamma_source_config_path', 
                default='../gamma_sources.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--accelerator_config_path', 
                default='../accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--plot_config_path', 
                default='../plot.yaml',
                type=click.Path(exists=True, dir_okay=False))

def main(gamma_source, accelerator, plot_type, gamma_cubes, obs_path, 
         hess_spectra_path, fermi_spectra_path,
         gamma_source_config_path, accelerator_config_path, plot_config_path):

    files = glob.glob(gamma_cubes)
    files = np.sort(files)
    
    with open(gamma_source_config_path) as file:
        gam_src_cfg = yaml.load(file)[gamma_source] 

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(plot_config_path) as file:
        plot_cfg = yaml.load(file)[plot_type]
    
    gamsrc_glon = gam_src_cfg['TeV']['glon']
    gamsrc_glat = gam_src_cfg['TeV']['glat']
    gamsrc_radius = gam_src_cfg['TeV']['radius']
    
    Emin = plot_cfg['Emin']
    Emax = plot_cfg['Emax']
    grid_size = plot_cfg['grid_size']
    radius = plot_cfg['radius']
    src_fits = plot_cfg['src_fits']
    con_levels = plot_cfg['con_levels']
    year = plot_cfg['year']
    glon_limit = plot_cfg['glon_limit']
    glat_limit = plot_cfg['glat_limit']
    snr_centre = plot_cfg['snr_centre']
    
    # HESS source     
    hess_spectra = np.loadtxt(hess_spectra_path, delimiter=',', skiprows=1)
    E_HESS = hess_spectra[:,0] * u.TeV
    Flux_HESS = hess_spectra[:,1] * 1/(u.TeV*u.cm**2*u.s)
    yerr_low_TeV = hess_spectra[:,2] * 1/(u.TeV*u.cm**2*u.s)
    yerr_high_TeV = hess_spectra[:,3] * 1/(u.TeV*u.cm**2*u.s)
    xerr_low_TeV = hess_spectra[:,4] * u.TeV
    xerr_high_TeV = hess_spectra[:,5] * u.TeV    
    sm_HESS = SpectralModel(E_HESS.value, Flux_HESS.value)
    
    # Fermi source
    fermi_spectra = np.loadtxt(fermi_spectra_path, delimiter=',', skiprows=1)
    E_Fermi = fermi_spectra[:,0] * u.TeV
    Flux_Fermi = fermi_spectra[:,1] * 1/(u.TeV*u.cm**2*u.s)
    yerr_low_GeV = fermi_spectra[:,2] * 1/(u.TeV*u.cm**2*u.s)
    yerr_high_GeV = fermi_spectra[:,3] * 1/(u.TeV*u.cm**2*u.s)
    xerr_low_GeV = fermi_spectra[:,4] * u.TeV
    xerr_high_GeV = fermi_spectra[:,5] * u.TeV
    sm_Fermi = SpectralModel(E_Fermi.value, Flux_Fermi.value)

    # Set up csv file
    output_filename = 'FluxParams.csv'
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Model', 'Age', 'Chi', 'Delta', 'D0 (cm-2 s-1)', 'n (cm-3)', 'n0 (cm-3)', 'ESN (erg)',
                         'Emax_t_ST0 (PeV)', 't_ST0 (yr)', 'Emax_t (TeV)', 'Deltap', 'Alpha', 'Eta', 
                         'Ebudget (erg)', 'Mej (solMass)', 'SN type', 
                         'Resc (pc)', 'Eesc (TeV)',
                         'chi2/ndf', 'Spatial std',
                         'Number of pixels', 'Number of non zero pixels', 'Bubble dominated',
                         'Plot_filename'])
    
    #------------------------------------ Model data ------------------------------------
    for f in files:
        figures = []
        cube = fits.open(f)
        data = cube[0].data
        header = cube[0].header
        
        E1 = eval(header['HENERGY1'])
        E2 = eval(header['HENERGY2'])
        
        base = os.path.splitext(f)[0]
        filename, newfilename = base.split('HadronicFlux_')    
        plt_filename = 'PlotOversample_'+newfilename+'.pdf'

        acc = Accelerator(
                name=acc_cfg['name'],
                glon=acc_cfg['glon'], 
                glat=acc_cfg['glat'], 
                radius=acc_cfg['radius'],
                age=header['HT_KYR']*u.kyr, 
                distance=acc_cfg['distance'],
                gas_mass=acc_cfg['gas_mass']
                ) 

        if (gam_src_cfg['TeV']['name'] != header['HGAM_SRC']) or \
            (acc.name != header['HACC'] or \
             plot_cfg['type'] != header['HPLTTYPE']):
            print('\nThe gamma-ray sources, accelerators or plot types dont match')
            print('\nTry to run again. Exiting')
            sys.exit()
        
        try:
            # Integrated flux
            sm3D = SpatialModel3D(data=data, header=header, energies=np.concatenate((E1, E2)).value)
            mask = sm3D.energies>Emin.value
            mask_i = int(np.argwhere(mask)[0])
            Esize, xsize, ysize = data.shape
            Energies = sm3D.energies[mask]
            
            integrated = np.empty((xsize, ysize))
            for ix in range(xsize):
                for iy in range(ysize):
                    if np.isfinite(data[mask_i:, ix, iy]).all():
                        integrated[ix, iy] = simps(data[mask_i:, ix, iy], Energies)
                    else:
                        integrated[ix, iy] = 0.0

            hdu = fits.PrimaryHDU(data=integrated)
            hdu.header = header
            hdu.header.remove('CTYPE3')
            hdu.header.remove('CUNIT3')
            hdu.header.remove('CRVAL3')
            hdu.header.remove('CRPIX3')
            hdu.header.remove('CDELT3')

            oversampled_header, oversampled_data = oversample(data=integrated, header=hdu.header, 
                                                              grid_size=grid_size, radius=radius)
            smI = SpatialModel(data=oversampled_data, header=oversampled_header)            

            # Goodness of fit 
            energy, flux = sm3D.sum_region(coords=[gamsrc_glon, gamsrc_glat], radius=gamsrc_radius)
            sm = SpectralModel(energy, flux)
            
            E = np.concatenate((E_Fermi, E_HESS))
            F = np.concatenate((Flux_Fermi, Flux_HESS))
            sigma  = np.concatenate((yerr_low_GeV, yerr_low_TeV))

            F_model = np.empty(len(F))
            for i in range(len(E)):
                F_model[i] = flux[np.argmin(abs(energy - E.value[i]))]

            residual = F.value - F_model
            ndf = len(residual)
            chi2_ndf = np.nansum( residual**2 / (sigma.value**2))  / ndf

            # Residuals
            obs_file = fits.open(obs_path)
            obs_data = obs_file[0].data
            obs_header = obs_file[0].header
            sm_obs = SpatialModel(data=obs_data, header=obs_header)
            residual_header, residual_data = smI.calc_residual(obs_data=obs_data, obs_header=obs_header, 
                                                         gamsrc_glon=gamsrc_glon, gamsrc_glat=gamsrc_glat, 
                                                         gamsrc_radius=gamsrc_radius)
            std_spatial = np.nanstd(residual_data.flatten())
            all_pixels = np.count_nonzero(~np.isnan(smI.obs_aperture))
            model_aperture = smI.model_aperture[~np.isnan(smI.model_aperture)]
            pixels_nonzero = len(model_aperture[model_aperture>0])

            # Bubble dominated or not
            wcs = WCS(hdu.header)
            py, px = wcs.wcs_world2pix(acc.glon, acc.glat, 0)
            inside_bubble = integrated[int(px), int(py)]

            Resc_deg = np.arctan(header['HRESC_PC']*u.pc / acc.distance).to(u.deg)
            position = SkyCoord(l=acc.glon, b=acc.glat, frame='galactic')
            aperture = SkyCircularAnnulus(position, Resc_deg, Resc_deg*1.1)
            mask = aperture.to_pixel(wcs).to_mask(method='center')
            mask_img = mask.multiply(integrated)
            mask_nonzero_img = mask_img[mask_img>0]
            outside_bubble = np.average(mask_nonzero_img)

            if outside_bubble < inside_bubble*1.1:
                bubble_dominated = 'True'
            else:
                bubble_dominated = 'False'


            # Save parameters to csv 
            with open(output_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                out = [header['HNAME']] + [header['HT_KYR']] \
                      + [header['HCHI']] + [header['HDELTA']] \
                      + [header['HD0_CM2S']] + [header['HN_CM3']] \
                      + [header['HN0_CM3']] + [header['HESN_ERG']] \
                      + [header['HEMT0_P']] + [header['HTST0_YR']] \
                      + [header['HEMT_TEV']] \
                      + [header['HDELTAP']] + [header['HALPHA']] + [header['HETA']] \
                      + [header['HEB_ERG']] + [header['HMEJ_SOL']] + [header['HSN_TYPE']] \
                      + [header['HRESC_PC']] + [header['HEES_TEV']] \
                      + [chi2_ndf] + [std_spatial] \
                      + [all_pixels] + [pixels_nonzero] + [bubble_dominated] \
                      + [plt_filename]
                writer.writerow(out)

            # Plotting and saving pdfs        
            figures.append(plt.figure(figsize=(8,5)))
            ax = figures[-1].add_subplot(111)
            sm.plot(ax=ax, fig=figures[-1], color='blue', ls='-', label=f'Model')
            sm_HESS.plot_errorbar(ax=ax, fmt='o', color='#4D4D4D', markersize=0,
                             yerr=[yerr_low_TeV.value, yerr_high_TeV.value], 
                             xerr=[xerr_low_TeV.value, xerr_high_TeV.value],
                             label='HESS J1804−216')
            sm_Fermi.plot_errorbar(ax=ax, fmt='o', color='#4D4D4D', alpha=0.6, markersize=0, 
                             yerr=[yerr_low_GeV.value, yerr_high_GeV.value], 
                             xerr=[xerr_low_GeV.value, xerr_high_GeV.value],
                             label='3FHL J1804.7−2144e')
            ax.set_ylim([1e-16, 5e-7])
            ax.set_xlim([1e-2, 5e1])
            ax.legend(loc='lower left')
            plt.close()

            figures.append(plt.figure(figsize=(8, 4)))
            ax = figures[-1].add_subplot(111, projection=WCS(oversampled_header))
            figures[-1], ax = smI.plot(oversampled_data, cmap=plt.cm.hot_r, ax=ax, fig=figures[-1],
                     colorbar_label=f'Integrated flux (>{Emin:.0f})'+' cm$^{-2}$ s$^{-1}$', 
                              diff_label=False)
            add_gamma_src(ax, src_fits, con_levels, year, colors='black')
            if snr_centre==True:
                add_SNR_centre(acc, ax, facecolor='blue')
            else:
                add_SNR(acc, ax, facecolor='None', edgecolor='blue', ls='--', lw=2)
            add_Resc(acc, header['HRESC_PC']*u.pc, ax, facecolor='None', edgecolor='#00CED1')
            set_limits(ax, oversampled_header, l=glon_limit, b=glat_limit)
            plt.close()

            figures.append(plt.figure(figsize=(11, 5.5)))
            ax = figures[-1].add_subplot(111, projection=WCS(residual_header))
            figures[-1], ax = smI.plot(residual_data, cmap=plt.cm.RdBu, ax=ax, fig=figures[-1], 
                              colorbar_label=f'Residual map', diff_label=False,
                              vmin=-7e-13, vmax=7e-13)
            add_gamma_src(ax, src_fits, con_levels, year, colors='black')
            plt.close()

            figures.append(plt.figure(figsize=(4,6)))
            ax = figures[-1].add_subplot(111)
            ax.text(0.02, 0.98, 'Model parameters', fontweight='bold', fontsize=17)
            ax.text(0.02, 0.01, f'Age={header["HT_KYR"]}kyr\nChi={header["HCHI"]}\nDelta={header["HDELTA"]} \
                                 \nD0={header["HD0_CM2S"]}cm$^2$s'+'$^{-1}'+f'$\nn={header["HN_CM3"]:.3f}cm'+'$^{-3}$' \
                                 + f'\nn0={header["HN0_CM3"]}cm'+'$^{-3}$'+f'\nESNR={header["HESN_ERG"]}erg \
                                 \nEmax_t_ST0={header["HEMT0_P"]}PeV\nt_ST0={header["HTST0_YR"]:.3f}yr \
                                 \nEmax_t={header["HEMT_TEV"]:.3f}TeV\
                                 \nDeltap={header["HDELTAP"]} \
                                 \nalpha={header["HALPHA"]}\neta={header["HETA"]:.3f}\nEbudget={header["HEB_ERG"]:.3e}erg \
                                 \nMej={header["HMEJ_SOL"]}solMass\nSN type={header["HSN_TYPE"]}\n \
                                 \nResc={Resc_deg.to(u.deg):.3f}\nEesc={header["HEES_TEV"]:.3f}TeV', fontsize=15)
            ax.text(1.1, 0.98, 'Flux parameters', fontweight='bold', fontsize=17)
            ax.text(1.1, 0.70, f'chi2/ndf={chi2_ndf:.3f} \nSpatial std={std_spatial:.3e} \
                               \nAll pixels={all_pixels} \
                               \nPixels nonzero={pixels_nonzero}\nBubble dominated={bubble_dominated}', 
                               fontsize=15)
            ax.set_axis_off()
            plt.close()


            with PdfPages(plt_filename) as pdf:
                for fig in figures:
                    pdf.savefig(fig, bbox_inches='tight')
                
        except:
            print('Error:', sys.exc_info()[0],f'\n for data file {f}')
        
if __name__ == '__main__':
    main()
