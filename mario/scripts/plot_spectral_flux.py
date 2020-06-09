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
@click.option('--hgps_catalog', 
                default='hgps_catalog_v1.fits',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('gamma_cube')

def main(gamma_source, gamma_cube, gamma_source_config_path, hgps_catalog):

    with open(gamma_source_config_path) as file:
        gam_src_cfg = yaml.load(file)[gamma_source]     

    gam_src = GammaSource(
        name=gam_src_cfg['TeV']['name'], 
        spectral_index=gam_src_cfg['TeV']['spectral_index'], 
        flux_E_GeV=gam_src_cfg['GeV']['integral_lower_energy'], 
        flux_GeV=gam_src_cfg['GeV']['integral_flux'], 
        flux_E_TeV=gam_src_cfg['TeV']['integral_lower_energy'], 
        flux_TeV=gam_src_cfg['TeV']['integral_flux'],
        )

    # HESS source with 2018 data
    cat = SourceCatalogHGPS(hgps_catalog)
    source = cat[gam_src.name]

    E_HESS = source.data.Flux_Points_Energy
    Flux_HESS = source.data.Flux_Points_Flux
    index_obs = source.data.Index_Spec_PL

    yerr_low = source.data.Flux_Points_Flux_Err_Lo
    yerr_high = source.data.Flux_Points_Flux_Err_Hi

    photon_flux = source.data.Flux_Spec_PL_Int_1TeV
    low_E = 1*u.TeV
    high_E = 1e20*u.TeV

    int_upper = ( (high_E.to(u.TeV)/u.TeV) ** (1 - index_obs) ) / (1 - index_obs)
    int_lower = ( (low_E.to(u.TeV)/u.TeV) ** (1 - index_obs) ) / (1 - index_obs)
    amplitude_obs = photon_flux / ((int_upper - int_lower)*u.TeV)

    # HESS source spectral model
    sm_HESS = SpectralModel(E_HESS.value, Flux_HESS.value)
    fit_obs = PL_log(np.log10(E_HESS.value), np.log10(amplitude_obs.value), index_obs)

    glon = source.data.GLON
    glat = source.data.GLAT
    radius = source.data.RSpec

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
    sm_HESS.plot_errorbar(fig=fig, ax=ax, yerr=[yerr_low.value, yerr_high.value], fmt='ro', 
                  label='Observations (2018)')
    ax.plot(E_HESS.value, 10**fit_obs, 'r')
    ax.set_ylim([1e-16, 1e-9])
    ax.set_xlim([1e-1, 5e1])
    ax.legend(loc='lower left')


    filename, newfilename = gamma_cube.split('Flux')
    fig.savefig(f'Spectral_flux'+newfilename+'.pdf', bbox_inches='tight', pad_inches=0.06)


if __name__ == '__main__':
    main()
