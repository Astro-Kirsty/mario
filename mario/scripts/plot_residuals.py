import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
#from astropy import coordinates as coord
from astropy.io import fits
from astropy.wcs import WCS
from gammapy.catalog import SourceCatalogHGPS

import click
from astropy.io.misc import yaml

from mario.sources.sources import Accelerator, GammaSource
from mario.gamma.spatialmodel import SpatialModel
from mario.plotting.plotting import add_SNR, add_SNR_centre, add_gamma_src, set_limits


@click.command()
@click.argument('accelerator')
@click.option('--accelerator_config_path', 
                default='accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('gamma_source')
@click.option('--gamma_source_config_path', 
                default='gamma_sources.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('plot_type')
@click.option('--plot_config_path', 
                default='plot.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--hgps_catalog', 
                default='hgps_catalog_v1.fits',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path')
@click.argument('obs_path')

def main(accelerator, gamma_source, plot_type, model_path, obs_path, hgps_catalog,
         accelerator_config_path, gamma_source_config_path, plot_config_path):

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(gamma_source_config_path) as file:
        gam_src_cfg = yaml.load(file)[gamma_source]     

    with open(plot_config_path) as file:
        plot_cfg = yaml.load(file)[plot_type]

    gam_src = GammaSource(
        name=gam_src_cfg['TeV']['name'], 
        spectral_index=gam_src_cfg['TeV']['spectral_index'], 
        flux_E_GeV=gam_src_cfg['GeV']['integral_lower_energy'], 
        flux_GeV=gam_src_cfg['GeV']['integral_flux'], 
        flux_E_TeV=gam_src_cfg['TeV']['integral_lower_energy'], 
        flux_TeV=gam_src_cfg['TeV']['integral_flux'],
        )

    for age in acc_cfg['age']:
        acc = Accelerator(
                name=acc_cfg['name'],
                glon=acc_cfg['glon'], 
                glat=acc_cfg['glat'], 
                radius=acc_cfg['radius'],
                age=age, 
                distance=acc_cfg['distance'],
                gas_mass=acc_cfg['gas_mass']
                )

    src_fits = plot_cfg['src_fits']
    con_levels = plot_cfg['con_levels']
    glon_limit = plot_cfg['glon_limit']
    glat_limit = plot_cfg['glat_limit']
    snr_centre = plot_cfg['snr_centre']


    # HESS source with 2018 data
    cat = SourceCatalogHGPS(hgps_catalog)
    source = cat[gam_src.name]
    glon = source.data.GLON
    glat = source.data.GLAT
    radius = source.data.RSpec

    # Observational data
    obs_file = fits.open(obs_path)
    obs_data = obs_file[0].data
    obs_header = obs_file[0].header
    sm_obs = SpatialModel(obs_data, obs_header)

    # Model data
    model_file = fits.open(model_path)
    model_data = model_file[0].data
    model_header = model_file[0].header
    sm_model = SpatialModel(model_data, model_header)
    residual_header, residual = sm_model.calc_residual(obs_data, obs_header, glon, glat, radius)


    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection=WCS(residual_header))
    fig, ax = sm_model.plot(residual, cmap=plt.cm.viridis, fig=fig, ax=ax, 
                      colorbar_label=f'Residual map', diff_label=True)

    add_gamma_src(src_fits, con_levels, ax, 'black')

    if snr_centre: 
        add_SNR_centre(acc, ax, facecolor='#00CED1')
    else:
        add_SNR(acc, ax, facecolor='None', edgecolor='blue', ls='--', lw=2)

    set_limits(ax, residual_header, l=glon_limit, b=glat_limit)

    filename, newfilename = model_path.split(f'_{accelerator}')
    fig.savefig(f'Residual_map_{accelerator}'+newfilename+'.pdf', bbox_inches='tight', pad_inches=0.06)

    plt.figure(figsize=(8, 4))
    plt.hist(residual.flatten(), bins=50, color='blue')
    plt.xlabel('Residuals')
    plt.savefig(f'Residual_hist_{accelerator}'+newfilename+'.pdf', bbox_inches='tight', pad_inches=0.06)

if __name__ == '__main__':
    main()
