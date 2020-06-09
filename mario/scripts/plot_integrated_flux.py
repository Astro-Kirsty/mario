import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy import coordinates as coord
from astropy.io import fits
from astropy.wcs import WCS

import click
from astropy.io.misc import yaml

from mario.sources.sources import Accelerator
from mario.gamma.spatialmodel import SpatialModel, SpatialModel3D, oversample
from mario.plotting.plotting import add_SNR, add_SNR_centre, add_gamma_src, set_limits


@click.command()
@click.argument('accelerator')
@click.option('--accelerator_config_path', 
                default='accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('plot_type')
@click.option('--plot_config_path', 
                default='plot.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('gamma_cube')

def main(accelerator, plot_type, gamma_cube, accelerator_config_path, plot_config_path):

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(plot_config_path) as file:
        plot_cfg = yaml.load(file)[plot_type]


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

    emin = plot_cfg['emin']
    emax = plot_cfg['emax']
    grid_size = plot_cfg['grid_size']
    radius = plot_cfg['radius']
    src_fits = plot_cfg['src_fits']
    con_levels = plot_cfg['con_levels']
    glon_limit = plot_cfg['glon_limit']
    glat_limit = plot_cfg['glat_limit']
    snr_centre = plot_cfg['snr_centre']

    cube = fits.open(gamma_cube)
    data = cube[0].data
    header = cube[0].header


    sm3D = SpatialModel3D(data, header)
    sm3D.fit('PL')

    integrated = sm3D.integrate_PL(emin, emax)


    hdu = fits.PrimaryHDU(data=integrated)
    hdu.header['CTYPE1'] = header['CTYPE1']
    hdu.header['CRVAL1'] = header['CRVAL1']
    hdu.header['CRPIX1'] = header['CRPIX1']
    hdu.header['CDELT1'] = header['CDELT1']

    hdu.header['CTYPE2'] = header['CTYPE2']
    hdu.header['CRVAL2'] = header['CRVAL2']
    hdu.header['CRPIX2'] = header['CRPIX2']
    hdu.header['CDELT2'] = header['CDELT2']

    hdu.header['GDELTA'] = header['GDELTA']
    hdu.header['GCHI'] = header['GCHI']
    hdu.header['GEBUDGET'] = header['GEBUDGET']


    oversampled_header, oversampled_data = oversample(data=integrated, header=hdu.header, 
                                                      grid_size=grid_size, radius=radius)

    sm = SpatialModel(oversampled_data, oversampled_header)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection=WCS(oversampled_header))
    fig, ax = sm.plot(oversampled_data, cmap=plt.cm.hot_r, fig=fig, ax=ax, 
                      colorbar_label=f'Integrated flux ({emin:.0f}-{emax:.0f}TeV) cm$^{-2}$ s$^{-1}$', 
                      diff_label=True)

    add_gamma_src(src_fits, con_levels, ax, 'black')

    if snr_centre: 
        add_SNR_centre(acc, ax, facecolor='#00CED1')
    else:
        add_SNR(acc, ax, facecolor='None', edgecolor='blue', ls='--', lw=2)

    set_limits(ax, oversampled_header, l=glon_limit, b=glat_limit)

    filename, newfilename = gamma_cube.split('Flux')
    fig.savefig(f'Integral_flux_{emin:.0f}_{emax:.0f}'+newfilename+'.pdf', bbox_inches='tight', pad_inches=0.06)


if __name__ == '__main__':
    main()
