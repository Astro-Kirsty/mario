import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import click
from astropy.io.misc import yaml

from mario.sources.sources import GammaSource
from mario.sources.sources import Accelerator
from mario.models.protonspectrum import ProtonSpectrum, TimeDependentProtonSpectrum
from mario.plotting.plotting import proton_spectrum, add_kcr_src, add_kcr


@click.command()
@click.argument('gamma_source')
@click.argument('accelerator')
@click.argument('model')
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
@click.option('--timedependent', default=False, type=bool)


def main(gamma_source, accelerator, model, output, timedependent, gamma_source_config_path, accelerator_config_path, model_config_path):

    with open(gamma_source_config_path) as file:
        gam_src_cfg = yaml.load(file)[gamma_source]

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(model_config_path) as file:
        model_cfg = yaml.load(file)[model]
      

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

    energies = np.logspace(0, 6, 50)*u.GeV
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
                        ps.flux(energies, rcloud=ps.dist_cloud)
                    else:
                        ps.flux(energies, rcloud=acc_cfg['distance_cloud_release_point'])
                        
                    if timedependent:
                        pst = TimeDependentProtonSpectrum(
                                gamma_src=gam_src, 
                                snr=acc, 
                                nH=acc_cfg['nH'],
                                diff_chi=chi, 
                                diff_delta=delta,
                                tsedov=model_cfg['tsedov'],
                                Ec=model_cfg['Ec'],
                                Ec_max=model_cfg['Ec_max'],
                                Ebudget=Ebudget,
                                n0=model_cfg['n0'], 
                                deltap=model_cfg['deltap'],
                                ) 
                        if 'cloud' in acc_cfg:
                            pst.define_cloud(cloud_coord=(
                                        acc_cfg['cloud']['glon'], 
                                        acc_cfg['cloud']['glat'])
                                        )
                            pst.flux(energies, rcloud=ps.dist_cloud)
                        else:
                            pst.flux(energies, rcloud=acc_cfg['distance_cloud_release_point'])
                        
                        figures[-1], ax  = proton_spectrum(
                                                spectrum=ps, 
                                                fig=figures[-1], ax=ax,
                                                ls=':')
                        figures[-1], ax  = proton_spectrum(
                                                spectrum=pst, 
                                                fig=figures[-1], ax=ax)
                    else:
                        figures[-1], ax  = proton_spectrum(
                            spectrum=ps, 
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

            ax = add_kcr_src(
                gamma_src=gam_src,
                snr=acc, 
                emin=gam_src_cfg['GeV']['plot_emin'], 
                emax=gam_src_cfg['GeV']['plot_emax'], 
                ax=ax, 
                color=gam_src_cfg['GeV']['color'], 
                ls='--', lw=2, zorder=0)
            ax = add_kcr_src(
                gamma_src=gam_src,
                snr=acc,
                emin=gam_src_cfg['TeV']['plot_emin'], 
                emax=gam_src_cfg['TeV']['plot_emax'], 
                ax=ax, 
                color=gam_src_cfg['TeV']['color'],
                ls='--', lw=2, zorder=0)

            ax = add_kcr(label='', 
                ax=ax,
                ls='-.', lw=3, alpha=0.5, zorder=0,
                color='magenta')

            lines = ax.get_lines()
            legend = plt.legend(
                lines[-3:],
                ['$k_{CR}$ (GeV)', '$k_{CR}$ (TeV)', '$k_{CR}$ (Earth)'],
                loc='lower left'
                )

    if timedependent:
        with PdfPages(output+f'TimeDependentProtonSpectra_{gam_src.name}_{acc.name}.pdf') as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
    else: 
        with PdfPages(output+f'ProtonSpectra_{gam_src.name}_{acc.name}.pdf') as pdf:
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')


if __name__ == '__main__':
    main()
