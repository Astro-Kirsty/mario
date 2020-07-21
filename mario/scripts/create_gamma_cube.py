import astropy.units as u
import numpy as np

import click
from astropy.io.misc import yaml

from mario.sources.sources import Accelerator
from mario.utils.gas_utils import GasParameters
from mario.gamma.flux import GammaFlux
from astropy.io import fits


@click.command()
@click.argument('accelerator')
@click.argument('total_column_density')
@click.argument('proton_cube')
@click.option('--accelerator_config_path', 
                default='accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', 
                help='Output path', default='./')


def main(accelerator_config_path, accelerator, total_column_density, proton_cube, output):

    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    file = fits.open(proton_cube)
    header = file[0].header 
    age, age_unit = header['GAGE'].split()

    acc = Accelerator(
                    name=acc_cfg['name'],
                    glon=acc_cfg['glon'], 
                    glat=acc_cfg['glat'], 
                    radius=acc_cfg['radius'],
                    age=float(age) * u.Unit(age_unit), 
                    distance=acc_cfg['distance'],
                    gas_mass=acc_cfg['gas_mass']
                    )

    Eg = np.logspace(np.log10(0.1), np.log10(100), 50)*u.TeV

    gp = GasParameters(snr=acc, gas_map_path=total_column_density)
    gs = GammaFlux(snr=acc, proton_cube_path=proton_cube, gp=gp)
    gs.flux(Eg)

    newfilename = proton_cube.split('ProtonFlux_')[1]
    gs.save_cube(filename='Flux_'+newfilename)

if __name__ == '__main__':
    main()
