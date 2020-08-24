import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import click
from astropy.io.misc import yaml

from mario.sources.sources import GammaSource
from mario.sources.sources import Accelerator
from mario.models.protonspectrum import ProtonSpectrum, TimeDependentProtonSpectrum
from mario.utils.gas_utils import GasParameters

from mario.utils.model_utils import calc_Bmag, calc_dist_CR_release_point, escape_time


@click.command()
@click.argument('gamma_source')
@click.argument('accelerator')
@click.argument('model')
@click.argument('total_column_density')
@click.option('--gamma_source_config_path', 
                default='gamma_sources.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--accelerator_config_path', 
                default='accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--model_config_path', 
                default='hadronic_models.yaml',
                type=click.Path(exists=True, dir_okay=False))

@click.option('-o', '--output', 
                help='Output path', default='./')


def main(gamma_source, accelerator, model, total_column_density, output, gamma_source_config_path, accelerator_config_path, model_config_path):

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
    timedependent = model_cfg['timedependent']

    Ep = np.logspace(0, 8, 2650)*u.GeV
    chis = model_cfg['diff_chi']
    deltas = model_cfg['diff_delta']
    Ebudgets = model_cfg['Ebudget']
    timedependent = model_cfg['timedependent']
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
                    if timedependent:
                        ps = TimeDependentProtonSpectrum(
                                gamma_src=gam_src, 
                                snr=acc, 
                                nH=acc_cfg['nH'],
                                diff_chi=chi, 
                                diff_delta=delta,
                                tsedov=model_cfg['tsedov'],
                                #Ec=model_cfg['Ec'],
                                Ec_max=model_cfg['Ec_max'],
                                n0=model_cfg['n0'], 
                                deltap=model_cfg['deltap'],
                                Ebudget=Ebudget
                                ) 

                        gp = GasParameters(snr=acc, gas_map_path=total_column_density)
                        tesc = escape_time(tsedov=ps.tsedov, Ec=Ep.to(u.TeV), Ec_max=ps.Ec_max, delta=ps.deltap)
                        Rc = calc_dist_CR_release_point(E51=ps.E51, n=ps.n0,
                                                       u1=ps.u1, tesc=tesc)

                        distance_from_release_point = np.empty((len(Rc), gp.dist_cell().shape[0], gp.dist_cell().shape[1]))
                        for i in range(len(Rc)):
                            distance_from_release_point[i] = gp.dist_cell().to(u.pc) - Rc[i]
                            distance_from_release_point[i][distance_from_release_point[i] < Rc[i].value] = 0

                        distance_from_release_point = distance_from_release_point * u.pc

                        ps.flux(Ep, rcloud=distance_from_release_point)
                        ps.save_cube(gp=gp, filename=f'ProtonFlux_Time_{accelerator}_age{acc.age.value:.0f}_delta{ps.diff_delta}_chi{ps.diff_chi}_E{ps.Ebudget.value}.fits')

                    else:
                        ps = ProtonSpectrum(
                                gamma_src=gam_src, 
                                snr=acc, 
                                nH=acc_cfg['nH'],
                                diff_chi=chi, 
                                diff_delta=delta,
                                tsedov=model_cfg['tsedov'],
                                Ec=model_cfg['Ec'],
                                Ec_max=model_cfg['Ec_max'],
                                deltap=model_cfg['deltap'],
                                n0=model_cfg['n0'],
                                Ebudget=Ebudget,
                                ) 

                        gp = GasParameters(snr=acc, gas_map_path=total_column_density)
                        tesc = escape_time(tsedov=ps.tsedov, Ec=ps.Ec, Ec_max=ps.Ec_max, delta=ps.deltap)
                        Rc = calc_dist_CR_release_point(E51=ps.E51, n=ps.n0,
                                                   u1=ps.u1, tesc=tesc)

                        distance_from_release_point = gp.dist_cell() - Rc
                        distance_from_release_point[distance_from_release_point < Rc] = 0

                        ps.flux(Ep, rcloud=distance_from_release_point)
                        ps.save_cube(gp=gp, filename=f'ProtonFlux_{accelerator}_age{acc.age.value:.0f}_delta{ps.diff_delta}_chi{ps.diff_chi}_E{ps.Ebudget.value}.fits')



if __name__ == '__main__':
    main()
