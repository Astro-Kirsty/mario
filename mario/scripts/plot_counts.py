import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits

from gammapy.data import DataStore
from gammapy.irf import load_cta_irfs

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
@click.argument('irf')
@click.option('--irf_config_path', 
                default='irf.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('gamma_cube')


def main(accelerator, plot_type, irf, gamma_cube, accelerator_config_path, plot_config_path, irf_config_path):

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(irf_config_path) as file:
        irf_cfg = yaml.load(file)[irf]

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

    src_fits = plot_cfg['src_fits']
    con_levels = plot_cfg['con_levels']
    glon_limit = plot_cfg['glon_limit']
    glat_limit = plot_cfg['glat_limit']
    snr_centre = plot_cfg['snr_centre']

    grid_size = irf_cfg['grid_size']
    radius = irf_cfg['radius']
    telescope = irf_cfg['telescope']
    irf_path = irf_cfg['irf_path']
    Emin = irf_cfg['Emin']
    Emax = irf_cfg['Emax']
    offset = irf_cfg['offset']
    livetime = irf_cfg['livetime']

    energy = np.logspace(np.log10(Emin.to(u.TeV).value), np.log10(Emax.to(u.TeV).value), 31)*u.TeV

    if telescope=='HESS':
        obs_id = irf_cfg['obs_id']
        hess_irf = DataStore.from_dir(irf_path)
        hess_obs = hess_irf.obs(obs_id)
        aeff = hess_obs.aeff.to_effective_area_table(offset=offset, energy=energy)
        edisp = hess_obs.edisp.to_energy_dispersion(offset=offset, e_true=energy, e_reco=energy)

    elif telescope=='CTA':
        cta_irf = load_cta_irfs(irf_path)
        aeff = cta_irf['aeff'].to_effective_area_table(offset=offset, energy=energy)
        edisp = cta_irf['edisp'].to_energy_dispersion(offset=offset, e_true=energy, e_reco=energy)


    file = fits.open(gamma_cube)
    cube = file[0].data
    cube_header = file[0].header

    sm3D = SpatialModel3D(cube, cube_header)
    sm3D.fit('best')
    sm3D.fake(aeff=aeff, edisp=edisp, livetime=livetime)
    gamma_counts = sm3D.counts

    hdu = fits.PrimaryHDU(data=gamma_counts)
    hdu.header['CTYPE1'] = cube_header['CTYPE1']
    hdu.header['CRVAL1'] = cube_header['CRVAL1']
    hdu.header['CRPIX1'] = cube_header['CRPIX1']
    hdu.header['CDELT1'] = cube_header['CDELT1']

    hdu.header['CTYPE2'] = cube_header['CTYPE2']
    hdu.header['CRVAL2'] = cube_header['CRVAL2']
    hdu.header['CRPIX2'] = cube_header['CRPIX2']
    hdu.header['CDELT2'] = cube_header['CDELT2']

    hdu.header['GDELTA'] = cube_header['GDELTA']
    hdu.header['GCHI'] = cube_header['GCHI']
    hdu.header['GEBUDGET'] = cube_header['GEBUDGET']

    oversampled_header, oversampled_data = oversample(data=gamma_counts, header=hdu.header, 
                                                      grid_size=grid_size, radius=radius)
    sm = SpatialModel(oversampled_data, oversampled_header)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection=WCS(oversampled_header))
    fig, ax = sm.plot(oversampled_data, cmap=plt.cm.hot_r, fig=fig, ax=ax,  
                      colorbar_label=f'{telescope} Counts', diff_label=True)

    ax.text(0.7, 0.03, f'Livetime={livetime}',
           horizontalalignment='left',
            transform=ax.transAxes,
            fontsize=12, color='black')

    add_gamma_src(src_fits, con_levels, ax, 'black')

    if snr_centre: 
        add_SNR_centre(acc, ax, facecolor='#00CED1')
    else:
        add_SNR(acc, ax, facecolor='None', edgecolor='blue', ls='--', lw=2)

    set_limits(ax, oversampled_header, l=glon_limit, b=glat_limit)

    filename, newfilename = gamma_cube.split('Flux_')
    fig.savefig(f'{telescope}_counts_'+newfilename+'.pdf', bbox_inches='tight', pad_inches=0.06)



if __name__ == '__main__':
    main()
