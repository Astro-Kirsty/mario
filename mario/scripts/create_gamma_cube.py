import astropy.units as u
import numpy as np

import click
from astropy.io.misc import yaml

from mario.sources.sources import GammaSource
from mario.sources.sources import Accelerator
from mario.models.protonspectrum import ProtonSpectrum
from mario.utils.gas_utils import GasParameters
from mario.gamma.flux import GammaFlux



@click.command()
@click.argument('gamma_source')
@click.argument('accelerator')
@click.argument('hadronic')
@click.argument('total_column_density')
@click.option('--gamma_source_config_path', 
                default='gamma_sources.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--accelerator_config_path', 
                default='accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--model_config_path', 
                default='models.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', 
                help='Output path', default='./')


def main(gamma_source_config_path, accelerator_config_path, model_config_path, accelerator, gamma_source, hadronic, total_column_density, output):

    with open(gamma_source_config_path) as file:
        gam_src_cfg = yaml.load(file)[gamma_source]

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(model_config_path) as file:
        model_cfg = yaml.load(file)[hadronic]
      

    gam_src = GammaSource(
        name=gam_src_cfg['TeV']['name'], 
        spectral_index=gam_src_cfg['TeV']['spectral_index'], 
        flux_E_GeV=gam_src_cfg['GeV']['integral_lower_energy'], 
        flux_GeV=gam_src_cfg['GeV']['integral_flux'], 
        flux_E_TeV=gam_src_cfg['TeV']['integral_lower_energy'], 
        flux_TeV=gam_src_cfg['TeV']['integral_flux'],
        )

    Ep = np.logspace(0, 6, 50)*u.GeV
    chis = model_cfg['diff_chi']
    deltas = model_cfg['diff_delta']
    Ebudgets = model_cfg['Ebudget']
    ages = model_cfg['acc_age']

    for age in ages:
        acc = Accelerator(
                name=acc_cfg['name'],
                glon=acc_cfg['glon'], 
                glat=acc_cfg['glat'], 
                radius=acc_cfg['radius'],
                age=age, 
                distance=acc_cfg['distance'],
                gas_mass=acc_cfg['gas_mass']
                )

        for chi in chis:
            for delta in deltas:
                for Ebudget in Ebudgets:
                    ps = ProtonSpectrum(
                        gamma_src=gam_src, 
                        snr=acc, 
                        nH=acc_cfg['nH'],
                        diff_chi=chi, 
                        diff_delta=delta,
                        tsedov=model_cfg['tsedov'],
                        Ec=model_cfg['Ec'],
                        Ec_max=model_cfg['Ec_max'],
                        Ebudget=Ebudget,
                        ) 

                    if 'cloud' in acc_cfg:
                        ps.define_cloud(cloud_coord=(
                                    acc_cfg['cloud']['glon'], 
                                    acc_cfg['cloud']['glat'])
                                    )
                        ps.flux(Ep, rcloud=ps.dist_cloud)
                    else:
                        ps.flux(Ep, rcloud=acc_cfg['distance_cloud_release_point'])

                    gp = GasParameters(snr=acc, gas_map_path=total_column_density)

                    Eg = np.logspace(np.log10(0.1), np.log10(100), 31)*u.TeV
                    gs = GammaFlux(snr=acc, ps=ps, gas=gp)
                    gs.flux(Eg)
                    gs.save_cube(filename=f'Flux_{accelerator}_age{acc.age.value:.0f}_delta{ps.diff_delta}_chi{ps.diff_chi}_E{ps.Ebudget.value}.fits')


if __name__ == '__main__':
    main()
