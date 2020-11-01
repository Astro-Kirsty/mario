import astropy.units as u
import numpy as np

import click
from astropy.io.misc import yaml

from mario.sources import Accelerator
from mario.models.hadronic_models import HadronicImpulsiveTimePDF, HadronicImpulsiveTimeExtentPDF
from mario.gas_utilities import GasParameters


@click.command()
@click.argument('accelerator')
@click.argument('model')
@click.argument('total_column_density')
@click.option('--accelerator_config_path', 
                default='accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--model_config_path', 
                default='hadronic_models.yaml',
                type=click.Path(exists=True, dir_okay=False))

@click.option('-o', '--output', 
                help='Output path', default='./')


def main(accelerator, model, total_column_density, output, accelerator_config_path, model_config_path):

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(model_config_path) as file:
        model_cfg = yaml.load(file)[model]
      
    chis = model_cfg['dif_chi']
    deltas = model_cfg['dif_delta']
    n0s = model_cfg['n0s']
    Emax_t_STs = model_cfg['Emax_t_STs']
    times = acc_cfg['age']
    Ep = np.logspace(0, 6, 2650)*u.GeV

    for t in times:
        acc = Accelerator(
                name=acc_cfg['name'],
                glon=acc_cfg['glon'], 
                glat=acc_cfg['glat'], 
                radius=acc_cfg['radius'],
                age=t,
                distance=acc_cfg['distance'],
                gas_mass=acc_cfg['gas_mass']
                )
        for chi in chis:
            for delta in deltas:
                for n0 in n0s:
                    for Emax_t_ST in Emax_t_STs:
                        ps = HadronicImpulsiveTimePDF(n=model_cfg['n'],
                                                      n0=n0,
                                                      t_ST=t,
                                                      dif_chi=chi,
                                                      dif_delta=delta,
                                                      Emax_t_ST=Emax_t_ST)
                        gp = GasParameters(snr=acc, gas_map_path=total_column_density)
                        R_vals = gp.dist_cell()
                        ps.calc_pdf(E=Ep, R=R_vals, t=t)
                        ps.save(author='Kirsty Feijen')
                                          


if __name__ == '__main__':
    main()
