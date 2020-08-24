import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import click
from astropy.io.misc import yaml

from mario.sources.sources import GammaSource
from mario.sources.sources import Accelerator
from mario.models.electronspectrum import ElectronSpectrum
from mario.plotting.plotting import electron_spectrum


@click.command()
@click.argument('gamma_source')
@click.argument('accelerator')
@click.option('--gamma_source_config_path', 
                default='gamma_sources.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--accelerator_config_path', 
                default='accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--model_config_path', 
                default='leptonic_models.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', 
                help='Output path', default='./')


def main(gamma_source_config_path, accelerator_config_path, model_config_path, gamma_source, accelerator, output):

    with open(gamma_source_config_path) as file:
        gam_src_cfg = yaml.load(file)[gamma_source]

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(model_config_path) as file:
        model_cfg = yaml.load(file)['hadronic']
      

    gam_src = GammaSource(
        name=gam_src_cfg['TeV']['name'], 
        spectral_index=gam_src_cfg['TeV']['spectral_index'], 
        flux_E_GeV=gam_src_cfg['GeV']['integral_lower_energy'], 
        flux_GeV=gam_src_cfg['GeV']['integral_flux'], 
        flux_E_TeV=gam_src_cfg['TeV']['integral_lower_energy'], 
        flux_TeV=gam_src_cfg['TeV']['integral_flux'],
        )

    chis = model_cfg['diff_chi']
    deltas = model_cfg['diff_delta']
    Ebudgets = model_cfg['Ebudget']

    energies = np.logspace(-3, 6, 50)*u.GeV
    figures = []

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

        for chi in chis:
            figures.append(plt.figure())
            ax = figures[-1].add_subplot(1, 1, 1)

            for delta in deltas:
                for Ebudget in Ebudgets:

                    es = ElectronSpectrum(
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
                        es.define_cloud(cloud_coord=(
                                    acc_cfg['cloud']['glon'], 
                                    acc_cfg['cloud']['glat'])
                                    )
                        es.flux(energies, rcloud=ps.dist_cloud)
                    else:
                        es.flux(energies, rcloud=acc_cfg['distance_cloud_release_point'])


                    figures[-1], ax  = electron_spectrum(
                        spectrum=es, 
                        fig=figures[-1], ax=ax)

            lines = ax.get_lines()
            legend = plt.legend(
                [lines[i] for i in range(len(deltas))],
                [f'$\delta$ = {j}' for j in deltas],
                loc = 'upper right'
                )
            ax.add_artist(legend)

            ax.text(0.05, 0.95, f'{acc_cfg["name"]} ({age:.0f})',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)

            ax.text(0.05, 0.88, f'$\chi$ = {chi}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)

    with PdfPages(output+f'ElectronSpectra_{gam_src.name}_{acc.name}.pdf') as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')


if __name__ == '__main__':
    main()
