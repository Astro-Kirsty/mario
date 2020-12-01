import numpy as np
import click
from astropy.io.misc import yaml
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import coordinates as coord
from photutils import SkyCircularAperture

from mario.sources import Accelerator
from mario.sources import GammaSource
from mario.gas_utilities import GasParameters
from mario.gamma.flux import GammaFlux
from mario.gamma.spatialmodel import SpatialModel, SpatialModel3D
from mario.gamma.spatialmodel import oversample



@click.command()
@click.argument('gamma_source')
@click.argument('accelerator')
@click.argument('plot_type')
@click.argument('total_column_density')
@click.argument('obs_path') # hgps map flux
@click.argument('pdf_cube')
@click.option('--accelerator_config_path', 
                default='accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--gamma_source_config_path', 
                default='gamma_sources.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--plot_config_path', 
                default='plot.yaml',
                type=click.Path(exists=True, dir_okay=False))

def main(gamma_source, accelerator, plot_type, total_column_density, obs_path, pdf_cube, 
         gamma_source_config_path, accelerator_config_path, plot_config_path):

    with open(gamma_source_config_path) as file:
        gam_src_cfg = yaml.load(file)[gamma_source] 

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(plot_config_path) as file:
        plot_cfg = yaml.load(file)[plot_type]
        
    acc = Accelerator(
                    name=acc_cfg['name'],
                    glon=acc_cfg['glon'], 
                    glat=acc_cfg['glat'], 
                    radius=acc_cfg['radius'], 
                    age=1*u.kyr, 
                    distance=acc_cfg['distance'],
                    gas_mass=acc_cfg['gas_mass']
                    )

    Eg = np.logspace(np.log10(0.1), np.log10(100), 100)*u.TeV

    alpha = 1.8

    gp = GasParameters(snr=acc, gas_map_path=total_column_density)
    gf = GammaFlux(snr=acc, pdf_cube_path=pdf_cube, gp=gp, alpha=alpha, eta=1)

    gf.flux(Eg)
    
    Emin = plot_cfg['Emin']
    Emax = plot_cfg['Emax']
    
    cube_header = gf.save_header()
    data = gf.fluxes.value
    sm3D = SpatialModel3D(data, cube_header)

    mask = sm3D.energies > 1
    i = int(np.argwhere(mask)[0])
    dEnergies = np.diff(sm3D.energies[mask])
    Esize, xsize, ysize = data.shape
    integrated = np.empty((xsize, ysize))
    for ix in range(xsize):
        for iy in range(ysize):
            integrated[ix, iy] = np.nansum(data[i:-1, ix, iy] * dEnergies)

    #sm3D.fit(fit='PL')
    #integrated = sm3D.integrate(Emin=1*u.TeV, Emax=100*u.TeV, fit='PL')
    
    hdu = fits.PrimaryHDU(data=integrated)
    hdu.header['BUNIT'] = cube_header['BUNIT']
    hdu.header['CTYPE1'] = cube_header['CTYPE1']
    hdu.header['CRVAL1'] = cube_header['CRVAL1']
    hdu.header['CRPIX1'] = cube_header['CRPIX1']
    hdu.header['CDELT1'] = cube_header['CDELT1']

    hdu.header['CTYPE2'] = cube_header['CTYPE2']
    hdu.header['CRVAL2'] = cube_header['CRVAL2']
    hdu.header['CRPIX2'] = cube_header['CRPIX2']
    hdu.header['CDELT2'] = cube_header['CDELT2']

    hdu.header['HNAME'] = cube_header['HNAME']
    hdu.header['HT_KYR'] = cube_header['HT_KYR']
    hdu.header['HCHI'] = cube_header['HCHI']
    hdu.header['HDELTA'] = cube_header['HDELTA']
    hdu.header['HD0_CM2S'] = cube_header['HD0_CM2S']
    hdu.header['HN_CM3'] = cube_header['HN_CM3']
    hdu.header['HN0_CM3'] = cube_header['HN0_CM3']
    hdu.header['HESN_ERG'] = cube_header['HESN_ERG']
    hdu.header['HEM100_P'] = cube_header['HEM100_P']
    hdu.header['HEMT_TEV'] = cube_header['HEMT_TEV']
    hdu.header['HTST_KYR'] = cube_header['HTST_KYR']
    hdu.header['HDELTAP'] = cube_header['HDELTAP']
    hdu.header['HEB_ERG'] = cube_header['HEB_ERG']
    hdu.header['HETA'] = cube_header['HETA']
    hdu.header['HALPHA'] = cube_header['HALPHA']
    hdu.header['HGASPATH'] = cube_header['HGASPATH']
    
    grid_size = plot_cfg['grid_size']
    radius = plot_cfg['radius']
    header3, oversample3 = oversample(integrated, hdu.header, grid_size=grid_size, radius=radius)
    sm = SpatialModel(oversample3, header3)
    
    file = fits.open(obs_path)
    obs_data = file[0].data
    obs_header = file[0].header
    smH = SpatialModel(obs_data, obs_header)
    
    model_wcs = WCS(sm.header)
    glon_last, glat_first = model_wcs.all_pix2world(0, 0, 0)
    glon_first, glat_last = model_wcs.all_pix2world(sm.data.shape[1], sm.data.shape[0], 0) 

    obs_wcs = WCS(obs_header)
    last_lon, last_lat = coord.SkyCoord(l=glon_last*u.deg, b=glat_last*u.deg, 
                                        frame='galactic').to_pixel(obs_wcs)
    first_lon, first_lat = coord.SkyCoord(l=glon_first*u.deg, b=glat_first*u.deg, 
                                          frame='galactic').to_pixel(obs_wcs)
    obs_crop = obs_data[int(first_lat):int(last_lat), int(last_lon):int(first_lon)]

    gamsrc_glon = gam_src_cfg['TeV']['glon']
    gamsrc_glat = gam_src_cfg['TeV']['glat']
    gamsrc_radius = gam_src_cfg['TeV']['radius']

    position = SkyCoord(l=gamsrc_glon, b=gamsrc_glat, frame='galactic')
    aperture = SkyCircularAperture(position, gamsrc_radius).to_pixel(model_wcs)
    mask = aperture.to_mask(method='center')

    obs_aperture = mask.multiply(obs_crop)
    model_aperture = mask.multiply(sm.data)

    
    factor = np.linspace(0.0001, 1, 100000)
    residual = []
    for f in factor:
        residual += [np.nansum(np.abs(obs_aperture - f * model_aperture))]
    eta = factor[np.argmin(residual)] 
    
    gf2 = GammaFlux(snr=acc, pdf_cube_path=pdf_cube, gp=gp, alpha=alpha, eta=eta)
    gf2.flux(Eg)
    gf2.save(author='Kirsty')


if __name__ == '__main__':
    main()
