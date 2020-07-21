import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy import coordinates as coord
from astropy.io import fits
from astropy.wcs import WCS
from gammapy.catalog import SourceCatalogHGPS

import click
from astropy.io.misc import yaml

from mario.sources.sources import GammaSource
from mario.gamma.spatialmodel import SpectralModel, PL_log, SpatialModel3D


@click.command()
@click.argument('gamma_source')
@click.option('--gamma_source_config_path', 
                default='gamma_sources.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('gamma_cube')
@click.argument('fermi_data')
@click.argument('hess_data')

def main(gamma_source, gamma_cube, fermi_data, hess_data, gamma_source_config_path):

    with open(gamma_source_config_path) as file:
        gam_src_cfg = yaml.load(file)[gamma_source]     

    # HESS source 
    hess = np.loadtxt(hess_data, delimiter=' ', comments='#')
    E_HESS = hess[:,0] * u.TeV                       
    Flux_HESS = hess[:,1] * 1 / (u.TeV * u.cm**2 * u.s) 
    yerr_low = hess[:,2] * 1 / (u.TeV * u.cm**2 * u.s)    
    yerr_high = hess[:,3] * 1 / (u.TeV * u.cm**2 * u.s)

    photon_flux = gam_src_cfg['TeV']['integral_flux']
    index_obs = gam_src_cfg['TeV']['spectral_index']
    low_E = gam_src_cfg['TeV']['integral_lower_energy']
    high_E = gam_src_cfg['TeV']['integral_higher_energy']

    # HESS source spectral model
    int_upper = ( (high_E.to(u.TeV)/u.TeV) ** (1 - index_obs) ) / (1 - index_obs)
    int_lower = ( (low_E.to(u.TeV)/u.TeV) ** (1 - index_obs) ) / (1 - index_obs)
    amplitude_obs = photon_flux / ((int_upper - int_lower)*u.TeV)
    sm_HESS = SpectralModel(E_HESS.value, Flux_HESS.value)
    fit_obs = PL_log(np.log10(E_HESS.value), np.log10(amplitude_obs.value), index_obs)

    # FGES source
    fges = np.loadtxt(fermi_data, delimiter=' ', comments='#')
    E_FGES = (fges[:,0]*u.GeV).to(u.TeV)
    E2_F_FGES = fges[:,1] * u.erg*1/(u.cm**2*u.s)
    FGES_yerr_low = (fges[:,2] * u.erg*1/(u.cm**2*u.s)).to(u.TeV*1/(u.cm**2*u.s)) / E_FGES**2
    FGES_yerr_high = (fges[:,3] * u.erg*1/(u.cm**2*u.s)).to(u.TeV*1/(u.cm**2*u.s)) / E_FGES**2
    Flux_FGES = (E2_F_FGES).to(u.TeV*1/(u.cm**2*u.s)) / E_FGES**2

    photon_flux_GeV = gam_src_cfg['GeV']['integral_flux']
    index_GeV = gam_src_cfg['GeV']['spectral_index']
    low_E_GeV = gam_src_cfg['GeV']['integral_lower_energy']
    high_E_GeV = gam_src_cfg['GeV']['integral_higher_energy']

    # FGES source spectral model
    int_upper_GeV = (high_E_GeV**(1-index_GeV)) / (1-index_GeV)
    int_lower_GeV = (low_E_GeV**(1-index_GeV)) / (1-index_GeV)
    F0_GeV = photon_flux_GeV / ((int_upper_GeV - int_lower_GeV)*u.TeV)
    sm_FGES = SpectralModel(E_FGES.value, Flux_FGES.value)
    fit_obs_FGES = PL_log(np.log10(E_FGES.value), np.log10(F0_GeV.value), index_GeV)


    glon = gam_src_cfg['TeV']['glon']
    glat = gam_src_cfg['TeV']['glat']
    radius = gam_src_cfg['TeV']['radius']

    # Model data
    cube = fits.open(gamma_cube)
    data = cube[0].data
    header = cube[0].header

    sm3D = SpatialModel3D(data, header)
    energy, flux = sm3D.sum_region(coords=[glon, glat], radius=radius)

    chi = header['GCHI']
    delta = header['GDELTA']
    Ebudget = header['GEBUDGET']

    sm = SpectralModel(energy, flux)
    amplitude, index = sm.fit('PL')
    fit_vals = sm.fit_data


    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    sm.plot(ax=ax, color='blue', label=f'Model $\chi$={chi}, $\delta$={delta}, '+'E$_{budget}$='+f'{Ebudget}')
    ax.plot(energy, 10**fit_vals, 'b')
    sm_HESS.plot_errorbar(ax=ax, yerr=[yerr_low.value, yerr_high.value], fmt='ro', 
                  label='HESS Observations (2004)')
    ax.plot(E_HESS.value, 10**fit_obs, 'r')
    sm_FGES.plot(ax=ax, color='green', label='FGES Observations')
    ax.plot(E_FGES.value, 10**fit_obs_FGES, 'g')
    ax.set_ylim([1e-16, 1e-7])
    ax.set_xlim([1e-2, 5e1])
    ax.legend(loc='lower left')


    filename, newfilename = gamma_cube.split('Flux_')
    fig.savefig(f'Spectral_Flux_'+newfilename+'.pdf', bbox_inches='tight', pad_inches=0.06)


if __name__ == '__main__':
    main()
