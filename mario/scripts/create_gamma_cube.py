import numpy as np
from astropy.io.misc import yaml
from astropy.io import fits
from astropy import units as u

from mario.sources import Accelerator
from mario.sources import GammaSource
from mario.gas_utilities import GasParameters
from mario.gamma.flux import GammaFlux
from mario.gamma.spatialmodel import SpatialModel3D

import sys
import click
import glob

@click.command()
@click.argument('gamma_source')
@click.argument('gamma_model')
@click.argument('accelerator')
@click.argument('plot_type')
@click.argument('pdf_cube')
@click.argument('taskid')

@click.option('--hess_spectra_path',
               default='/home/kirsty/Documents/PhD/GitHub/data/HESS_spectra_data.csv',
               type=click.Path(exists=True, dir_okay=False))
@click.option('--fermi_spectra_path',
               default='/home/kirsty/Documents/PhD/GitHub/data/Fermi_spectra_data.csv',
               type=click.Path(exists=True, dir_okay=False))
@click.option('--gamma_source_config_path', 
                default='/home/kirsty/Documents/PhD/GitHub/mario/examples/gamma_sources.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--gamma_model_config_path', 
                default='/home/kirsty/Documents/PhD/GitHub/mario/examples/gamma_models.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--accelerator_config_path', 
                default='/home/kirsty/Documents/PhD/GitHub/mario/examples/accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--obs_path', 
               default='/home/kirsty/Documents/PhD/Fits_Files/Models/hgps_map_flux_0.1deg_v1_crop.fits',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--plot_config_path', 
                default='/home/kirsty/Documents/PhD/GitHub/mario/examples/plot.yaml',
                type=click.Path(exists=True, dir_okay=False))

def main(gamma_source, gamma_model, accelerator, plot_type, pdf_cube, taskid, 
		 hess_spectra_path, fermi_spectra_path,
         gamma_source_config_path, gamma_model_config_path, accelerator_config_path, obs_path, plot_config_path):

    with open(gamma_source_config_path) as file:
        gam_src_cfg = yaml.load(file)[gamma_source] 
        
    with open(gamma_model_config_path) as file:
        gam_cfg = yaml.load(file)[gamma_model] 

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(plot_config_path) as file:
        plot_cfg = yaml.load(file)[plot_type]

    hess_spectra = np.loadtxt(hess_spectra_path, delimiter=',', skiprows=1)
    E_HESS = hess_spectra[:,0] * u.TeV
    Flux_HESS = hess_spectra[:,1] * 1/(u.TeV*u.cm**2*u.s)

    fermi_spectra = np.loadtxt(fermi_spectra_path, delimiter=',', skiprows=1)
    E_Fermi = fermi_spectra[:,0] * u.TeV
    Flux_Fermi = fermi_spectra[:,1] * 1/(u.TeV*u.cm**2*u.s)
    
    gamsrc_glon = gam_src_cfg['TeV']['glon']
    gamsrc_glat = gam_src_cfg['TeV']['glat']
    gamsrc_radius = gam_src_cfg['TeV']['radius']
    
    alpha = gam_cfg['alpha']
    xsection = gam_cfg['xsection']
    depth = gam_cfg['depth']

    files = glob.glob(pdf_cube)
    files = np.sort(files)
    cube = fits.open(files[int(taskid)])
    header = cube[0].header
    total_column_density = header['HGASPATH']

    acc = Accelerator(
                    name=acc_cfg['name'],
                    glon=acc_cfg['glon'], 
                    glat=acc_cfg['glat'], 
                    radius=acc_cfg['radius'], 
                    age=1*u.kyr, 
                    distance=acc_cfg['distance'],
                    gas_mass=acc_cfg['gas_mass']
                    )

    if (gam_src_cfg['TeV']['name'] != header['HGAM_SRC']) or (acc.name != header['HACC']):
        print('\nThe gamma-ray sources or accelerators dont match')
        print('Try to run again. Exiting')
        sys.exit()
   
    gp = GasParameters(snr=acc, gas_map_path=total_column_density)
   
    E = np.concatenate((E_Fermi.to(u.TeV), E_HESS.to(u.TeV)))
    F = np.concatenate((Flux_Fermi, Flux_HESS))
    gf = GammaFlux(acc=acc, pdf_cube_path=files[int(taskid)], gp=gp, alpha=alpha, eta=1, 
                   xsection=xsection, depth=depth)

    gf.flux(np.array(E)*u.TeV)
    extra_comments = {'HGAM_SRC':header['HGAM_SRC'],
                      'HACC':header['HACC']}
    cube_header = gf.save_header(extra_comments=extra_comments)
    data = gf.fluxes.value
    
    sm3D = SpatialModel3D(data, cube_header)   
    e, F_model = sm3D.sum_region(coords=[gamsrc_glon, gamsrc_glat], radius=gamsrc_radius)

    factor = np.linspace(0.0001, 1, 100000)
    residual = []
    for f in factor:
        residual += [np.nansum(np.abs(F.value - f * F_model))]
    eta = factor[np.argmin(residual)] 

    E1 = np.logspace(-2, 0, 100)[:-1]*u.TeV
    E2 = np.logspace(0, 2, 100)*u.TeV
    Eg = np.concatenate((E1, E2))
    gf2 = GammaFlux(acc=acc, pdf_cube_path=files[int(taskid)], gp=gp, alpha=alpha, eta=eta, 
                    xsection=xsection, depth=depth)
    gf2.flux(np.array(Eg)*u.TeV)
    extra_comments = {'HGAM_SRC':header['HGAM_SRC'],
                      'HACC':header['HACC'],
                      'HPLTTYPE':plot_cfg['type'],
                      'HENERGY1':'np.logspace(-2, 0, 100)[:-1]*u.TeV',
                      'HENERGY2':'np.logspace(0, 2, 100)*u.TeV'}
    gf2.save(author='Kirsty Feijen', extra_comments=extra_comments)


if __name__ == '__main__':
    main()
