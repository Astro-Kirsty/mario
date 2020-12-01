import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mario.models.utilities import calc_shock_radius
from matplotlib.patches import Circle

import astropy.units as u
from astropy import coordinates as coord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io.misc import yaml

import os
import click
import glob
import csv

from mario.sources import GammaSource, Accelerator
from mario.gamma.spatialmodel import SpectralModel, SpatialModel3D, SpatialModel
from mario.gamma.spatialmodel import oversample, PL_log
from mario.plotting import add_SNR, add_gamma_src, set_limits

@click.command()
@click.argument('gamma_source')
@click.argument('accelerator')
@click.argument('plot_type')
@click.argument('hess_spectra')
@click.argument('obs_path')
@click.argument('gamma_cubes')
@click.option('--gamma_source_config_path', 
                default='gamma_sources.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--accelerator_config_path', 
                default='accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--plot_config_path', 
                default='plot.yaml',
                type=click.Path(exists=True, dir_okay=False))


def main(gamma_source, accelerator, plot_type, hess_spectra, obs_path, gamma_cubes, 
         gamma_source_config_path, accelerator_config_path, plot_config_path):

    files = glob.glob(gamma_cubes)
    files = np.sort(files)
    
    with open(gamma_source_config_path) as file:
        gam_src_cfg = yaml.load(file)[gamma_source] 

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(plot_config_path) as file:
        plot_cfg = yaml.load(file)[plot_type] 
        
    Emin = plot_cfg['Emin']
    Emax = plot_cfg['Emax']
    fit_type = plot_cfg['fit_type']
    grid_size = plot_cfg['grid_size']
    radius = plot_cfg['radius']
    src_fits = plot_cfg['src_fits']
    con_levels = plot_cfg['con_levels']
    glon_limit = plot_cfg['glon_limit']
    glat_limit = plot_cfg['glat_limit']

    # HESS source 
    hess = np.loadtxt(hess_spectra, delimiter=' ', comments='#')
    E_HESS = hess[:,0] * u.TeV                       
    Flux_HESS = hess[:,1] * 1 / (u.TeV * u.cm**2 * u.s) 
    yerr_low = hess[:,2] * 1 / (u.TeV * u.cm**2 * u.s)    
    yerr_high = hess[:,3] * 1 / (u.TeV * u.cm**2 * u.s)
    bin_w = 0.5 * np.diff(np.log10(E_HESS.value))
    bin_ww = np.append([0.05], bin_w)
    xerr_low = E_HESS.value - 10**(np.log10(E_HESS.value) - bin_ww)
    xerr_high = 10**(np.log10(E_HESS.value) + bin_ww) - E_HESS.value

    sm_HESS = SpectralModel(E_HESS.value, Flux_HESS.value)

    gamsrc_glon = gam_src_cfg['TeV']['glon']
    gamsrc_glat = gam_src_cfg['TeV']['glat']
    gamsrc_radius = gam_src_cfg['TeV']['radius']
     
    output_filename = 'FluxParams.csv'
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Model', 'Age', 'Chi', 'Delta', 'D0 (cm-2 s-1)', 'n (cm-3)', 'n0 (cm-3)', 'ESN (erg)',
                         'Emax_100 (PeV)', 'Emax_t_ST (TeV)', 't_ST (kyr)', 'Deltap', 'Alpha', 'Eta', 
                         'Ebudget (erg)', 'Resc (pc)', 'Eesc (TeV)',
                         'Amplitude (cm-2 s-1 TeV-1)', 'Index', 'Lambda', 'Alpha_spec', 
                         'Integrated flux', 'Residual sum (Integrated)',
                         'Chi2 (uncertainty)', 'Chi2 (actual value)', 'Residual sum (spectral)'])
    
    #------------------------------------ Model data ------------------------------------
    for f in files:
        figures = []
        cube = fits.open(f)
        data = cube[0].data
        header = cube[0].header

        acc = Accelerator(
                name=acc_cfg['name'],
                glon=acc_cfg['glon'], 
                glat=acc_cfg['glat'], 
                radius=acc_cfg['radius'],
                age=header['HTST_KYR']*u.kyr, 
                distance=acc_cfg['distance'],
                gas_mass=acc_cfg['gas_mass']
                ) 

        #-------------------------------- Integrated flux --------------------------------
        sm3D = SpatialModel3D(data=data, header=header)
        sm3D.fit(fit='ECO')
        energy, flux = sm3D.sum_region(coords=[gamsrc_glon, gamsrc_glat], radius=gamsrc_radius)
        sm = SpectralModel(energy, flux)
        sm.fit(fit='ECO')
        amplitude, index, _lambda, alpha_spec = sm.popt
        mask2 = sm.energies > Emin.value
        j = int(np.argwhere(mask2)[0])
        integrated_flux_val = np.nansum(sm.fluxes[j:-1] * np.diff(sm.energies[mask2]))

#        mask = sm3D.energies>Emin.value
#        i = int(np.argwhere(mask)[0])
#        dEnergies = np.diff(sm3D.energies[mask])
#        Esize, xsize, ysize = data.shape
#        integrated = np.empty((xsize, ysize))
#        for ix in range(xsize):
#            for iy in range(ysize):
#                integrated[ix, iy] = np.nansum(data[i:-1, ix, iy] * dEnergies)

        integrated = sm3D.integrate(Emin=1*u.TeV, Emax=100*u.TeV, fit='ECO')

        hdu = fits.PrimaryHDU(data=integrated)
        hdu.header['BUNIT'] = header['BUNIT']
        hdu.header['CTYPE1'] = header['CTYPE1']
        hdu.header['CRVAL1'] = header['CRVAL1']
        hdu.header['CRPIX1'] = header['CRPIX1']
        hdu.header['CDELT1'] = header['CDELT1']

        hdu.header['CTYPE2'] = header['CTYPE2']
        hdu.header['CRVAL2'] = header['CRVAL2']
        hdu.header['CRPIX2'] = header['CRPIX2']
        hdu.header['CDELT2'] = header['CDELT2']

        hdu.header['HNAME']   = header['HNAME']
        hdu.header['HT_KYR']  = header['HT_KYR']
        hdu.header['HCHI']    = header['HCHI']
        hdu.header['HDELTA']  = header['HDELTA']
        hdu.header['HD0_CM2S'] = header['HD0_CM2S']
        hdu.header['HN_CM3']  = header['HN_CM3']
        hdu.header['HN0_CM3'] = header['HN0_CM3']
        hdu.header['HESN_ERG'] = header['HESN_ERG']
        hdu.header['HEM100_P']  = header['HEM100_P']
        hdu.header['HEMT_TEV'] = header['HEMT_TEV']
        hdu.header['HTST_KYR'] = header['HTST_KYR']
        hdu.header['HDELTAP'] = header['HDELTAP']
        hdu.header['HEB_ERG'] = header['HEB_ERG']

        oversampled_header, oversampled_data = oversample(data=integrated, header=hdu.header, 
                                                          grid_size=grid_size, radius=radius)

        smI = SpatialModel(data=oversampled_data, header=oversampled_header)
        
        #------------------------------------- Goodness of fit ----------------------------------
        # chi2_unc = np.sum((obs - fit) ** 2 / unc**2)
        # chi2_actual = np.sum((obs - fit) ** 2 / obs)
        amplitude = sm.popt[0]
        index = sm.popt[1]
        _lambda = sm.popt[2]
        alpha = sm.popt[3]
        fit_data = amplitude * E_HESS.value **(-index) * np.exp(-(_lambda * E_HESS.value)**alpha)
        residual = Flux_HESS.value - fit_data
        residual = np.array(residual)
        res_sum_spec = np.nansum(residual)
        chi2_unc = np.nansum( residual**2 / (yerr_high.value**2))
        chi2_act = np.nansum( residual**2 / Flux_HESS.value)

        #--------------------------------------- Residual ---------------------------------------
        # Integrated flux observational data
        obs_file = fits.open(obs_path)
        obs_data = obs_file[0].data
        obs_header = obs_file[0].header
        sm_obs = SpatialModel(data=obs_data, header=obs_header)
        residual_header, residual = smI.calc_residual(obs_data=obs_data, obs_header=obs_header, 
                                                     gamsrc_glon=gamsrc_glon, gamsrc_glat=gamsrc_glat, 
                                                     gamsrc_radius=gamsrc_radius)
        res_sum = np.nansum(np.absolute(residual))

        #-------------------------------- Save parameters to file --------------------------------
        with open(output_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            out = [header['HNAME']] + [header['HT_KYR']] \
                  + [header['HCHI']] + [header['HDELTA']] \
                  + [header['HD0_CM2S']] + [header['HN_CM3']] \
                  + [header['HN0_CM3']] + [header['HESN_ERG']] \
                  + [header['HEM100_P']] + [header['HEMT_TEV']] + [header['HTST_KYR']] \
                  + [header['HDELTAP']] + [header['HALPHA']] + [header['HETA']] \
                  + [header['HEB_ERG']] + [header['HRESC_PC']] + [header['HEES_TEV']] \
                  + [amplitude] + [index] + [_lambda] + [alpha_spec] \
                  + [integrated_flux_val] + [res_sum] \
                  + [res_sum_spec] + [chi2_unc] + [chi2_act]
            writer.writerow(out)
            
        base = os.path.splitext(f)[0]
        filename, newfilename = base.split('HadronicFlux_')
        
        #--------------------------- Plotting and saving pdfs and fits ---------------------------        
        figures.append(plt.figure(figsize=(8,5)))
        ax = figures[-1].add_subplot(111)
        sm.plot(ax=ax, fig=figures[-1], color='blue', ls='-', label=f'Model')
        sm_HESS.plot_errorbar(ax=ax, yerr=[yerr_low.value, yerr_high.value], markersize=0, 
                              xerr=[xerr_low, xerr_high], fmt='ro',
                              label='HESS Observations (2004)')
        ax.set_ylim([1e-16, 1e-9])
        ax.set_xlim([5e-2, 5e1])
        ax.legend(loc='lower left')
        plt.close()
        
        figures.append(plt.figure(figsize=(8, 4)))
#        ax = figures[-1].add_subplot(111, projection=WCS(hdu.header))
        ax = figures[-1].add_subplot(111, projection=WCS(oversampled_header))
#        figures[-1], ax = smI.plot(integrated, cmap=plt.cm.hot_r, ax=ax, fig=figures[-1],
        figures[-1], ax = smI.plot(oversampled_data, cmap=plt.cm.hot_r, ax=ax, fig=figures[-1],
                 colorbar_label=f'Integrated flux (>{Emin:.0f})'+' cm$^{-2}$ s$^{-1}$', 
                          diff_label=False)
        add_gamma_src(src_fits, con_levels, ax, colors='black')
        add_SNR(acc, ax, facecolor='None', edgecolor='blue', ls='--', lw=2)
        Resc = np.arctan(header['HRESC_PC']*u.pc / acc.distance)
        snr_region = Circle((acc.glon.value, acc.glat.value), Resc.to(u.deg).value,
                        transform=ax.get_transform('galactic'), facecolor='None', edgecolor='#00CED1')
        ax.add_patch(snr_region)

#        set_limits(ax, hdu.header, l=glon_limit, b=glat_limit)
        set_limits(ax, oversampled_header, l=glon_limit, b=glat_limit)
        #smI.save_fits(data=oversampled_data, filename=f'IntegralFlux_{fit_type}_E{Emin:.0f}-{Emax:.0f}_'+newfilename+'.fits')
        plt.close()

        figures.append(plt.figure(figsize=(11, 5.5)))
        ax = figures[-1].add_subplot(111, projection=WCS(residual_header))
        figures[-1], ax = smI.plot(residual, cmap=plt.cm.RdBu, ax=ax, fig=figures[-1], 
                          colorbar_label=f'Residual map', diff_label=False,
                          vmin=-7e-13, vmax=7e-13)
        add_gamma_src(src_fits, con_levels, ax, colors='black', alpha=0.5)
        plt.close()

        figures.append(plt.figure(figsize=(8,6)))
        ax = figures[-1].add_subplot(111)
        ax.text(0.02, 0.9, 'Model parameters', fontweight='bold', fontsize=17)
        ax.text(0.02, 0.02, f'Age={header["HT_KYR"]}kyr\nChi={header["HCHI"]}\nDelta={header["HDELTA"]} \
                             \nD0={header["HD0_CM2S"]}cm$^2$s'+'$^{-1}'+f'$\nn={header["HN_CM3"]}cm'+'$^{-3}$' \
                             + f'\nn0={header["HN0_CM3"]}cm'+'$^{-3}$'+f'\nESNR={header["HESN_ERG"]}erg \
                             \nEmax_100={header["HEM100_P"]}PeV\nEmax_t={header["HEMT_TEV"]}TeV \
                             \nT_st={header["HTST_KYR"]}kyr\nDeltap={header["HDELTAP"]} \
                             \nalpha={header["HALPHA"]}\neta={header["HETA"]}\nEbudget={header["HEB_ERG"]}erg\n \
                             \nResc={Resc.to(u.deg)}\nEesc={header["HEES_TEV"]}TeV', fontsize=15)
        ax.text(0.7, 0.9, 'Flux parameters', fontweight='bold', fontsize=17)
        ax.text(0.7, 0.4, f'Amplitude={amplitude:.4e}\nIndex={index}\nLambda={_lambda},\nAlpha={alpha_spec} \
                           \nIntegrated flux val={integrated_flux_val}\nResidual sum (integrated)={res_sum} \
                           \nResidual sum (spectra)={res_sum_spec}\nChi2 (uncertainty)={chi2_unc} \
                           \nChi2 (actual value)={chi2_act}', 
                           fontsize=15)
        ax.set_axis_off()
        plt.close()
                       
        with PdfPages('PlotOversample_'+newfilename+'.pdf') as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
        
if __name__ == '__main__':
    main()
