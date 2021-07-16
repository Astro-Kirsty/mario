"""
Example script to create a PDF cube in the hadronic scenario. To run:
python create_pdf_cube.py 'G8_15' 'HESSJ1804' 's1_G8_15' ../data/total_column_density_mock.fits

This example produces the PDF cube for protons in the energy range from 0.1 to Emax GeV.
The accelerator and gamma-ray source models are defined by the yaml files. You can adjust these to be correct for your own accelerator and gamma-ray source of interest. 
The hadronic models to run are defined in the hadronic_models yaml (which can be created through the create_hadronic_yaml.py), for Type Ia and Type II SN. 
"""


import astropy.units as u
import numpy as np

import click
from astropy.io.misc import yaml

from mario.sources import Accelerator, GammaSource
from mario.models.hadronic_models import HadronicImpulsiveTimePDF, HadronicImpulsiveTimeExtentPDF
from mario.gas_utilities import GasParameters, calc_average_number_density


@click.command()
@click.argument('accelerator')
@click.argument('gamma_source')
@click.argument('model')
@click.argument('total_column_density')
@click.option('--accelerator_config_path', 
                default='../accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--gamma_source_config_path', 
                default='../gamma_sources.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--model_config_path', 
                default='../hadronic_models.yaml',
                type=click.Path(exists=True, dir_okay=False))

@click.option('-o', '--output', 
                help='Output path', default='./')


def main(accelerator, gamma_source, model, total_column_density, output, 
         accelerator_config_path, gamma_source_config_path, model_config_path):

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]
        
    with open(gamma_source_config_path) as file:
        gam_cfg = yaml.load(file)[gamma_source]

    with open(model_config_path) as file:
        model_cfg = yaml.load(file)[model]
      
    chis = model_cfg['dif_chi']
    deltas = model_cfg['dif_delta']
    n0s = model_cfg['n0s']
    deltaps = model_cfg['deltaps']
    Mejs = model_cfg['Mejs']
    Emax_t_ST0s = model_cfg['Emax_t_ST0s']
    
    gam_src = GammaSource(name=gam_cfg['TeV']['name'], 
                          spectral_index=gam_cfg['TeV']['spectral_index'], 
                          glon=gam_cfg['TeV']['glon'], glat=gam_cfg['TeV']['glat'], 
                          radius=gam_cfg['TeV']['radius'],
                          flux_GeV=gam_cfg['GeV']['integral_flux'],
                          flux_E_GeV=gam_cfg['GeV']['integral_lower_energy'],
                          flux_TeV=gam_cfg['TeV']['integral_flux'],
                          flux_E_TeV=gam_cfg['TeV']['integral_lower_energy'])

    acc = Accelerator(
            name=acc_cfg['name'],
            glon=acc_cfg['glon'], 
            glat=acc_cfg['glat'], 
            radius=acc_cfg['radius'],
            age=acc_cfg['age'],
            distance=acc_cfg['distance'],
            gas_mass=acc_cfg['gas_mass']
            )
    gp = GasParameters(acc=acc, gas_map_path=total_column_density)
    R_vals = gp.dist_cell()
    
    n = calc_average_number_density(total_column_density, gam_src, acc)

    for chi in chis:
        for delta in deltas:
            for n0 in n0s:
                for deltap in deltaps:
                    for Mej in Mejs:
                        for Emax_t_ST0 in Emax_t_ST0s:
                            ps = HadronicImpulsiveTimePDF(n=n,
                                                          n0=n0,
                                                          dif_chi=chi,
                                                          dif_delta=delta,
                                                          deltap=deltap,
                                                          Emax_t_ST0=Emax_t_ST0,
                                                          Mej=Mej,
                                                          E_SN=model_cfg['E_SN'])
                            Ep = np.logspace(0, np.log10(Emax_t_ST0.to(u.GeV).value), 1000)*u.GeV
                            ps.calc_pdf(E=Ep, R=R_vals, t=acc.age)
                            extra_comments = {'HGAM_SRC':gam_src.name,
                                              'HACC':acc.name,
                                              'HGASPATH':total_column_density}
                            ps.save(author='Kirsty Feijen', extra_comments=extra_comments)
                                          


if __name__ == '__main__':
    main()
