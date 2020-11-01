import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c
from matplotlib.backends.backend_pdf import PdfPages

import click
from astropy.io.misc import yaml

from mario.sources import GammaSource
from mario.sources import Accelerator
from mario.models.hadronic_models import HadronicImpulsivePDF, HadronicImpulsiveTimePDF, HadronicImpulsiveTimeExtentPDF
from mario.plotting import proton_spectrum, add_kcr_src, add_kcr


@click.command()
@click.argument('accelerator')
@click.argument('model')
@click.option('--accelerator_config_path', 
                default='accelerators.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--model_config_path', 
                default='hadronic_models.yaml',
                type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', 
                help='Output path', default='./')

def main(accelerator, model, output, accelerator_config_path, model_config_path):


    with open(accelerator_config_path) as file:
        acc_cfg = yaml.load(file)[accelerator]

    with open(model_config_path) as file:
        model_cfg = yaml.load(file)[model]

      
    chis = model_cfg['dif_chi']
    deltas = model_cfg['dif_delta']
    n0s = model_cfg['n0s']
    etas = model_cfg['etas']
    alpha = model_cfg['alpha']
    times = acc_cfg['age']
    Ep = np.logspace(0, 6, 2650)*u.GeV
    figures = []

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
        for n0 in n0s:
            for eta in etas:
                for chi in chis:
                    figures.append(plt.figure())
                    ax = figures[-1].add_subplot(1, 1, 1)
                    lines = []
                    for delta in deltas:
                        for Emax_t_ST in Emax_t_STs:

                            ps = HadronicImpulsiveTimePDF(n=model_cfg['n'],
                                                          n0=n0,
                                                          t_ST=t,
                                                          dif_chi=chi,
                                                          dif_delta=delta,
                                                          Emax_t_ST=Emax_t_ST)
                            ps.calc_pdf(E=Ep, R=[50]*u.pc, t=t)
    #                        ps.calc_pdf(E=Ep, R=model_cfg['R'], t=t)
                            Ebudget = eta * ps.E_SN
                            Emin = 1*u.GeV
                            Emax = Emax_t_ST
                            N0 = (Ebudget * (1-alpha) / (((Emax/u.GeV)**(1-alpha) - (Emin/u.GeV)**(1-alpha))*u.GeV)).decompose()

                            cool_time = (6e7 * (ps.n / u.cm**-3) ** (-1) * u.yr)
                            losses = np.exp(- ((alpha - 1) * t  / cool_time)) 

                            # proton spectra for one value of R
                            flux = (ps.pdf[:,0,0] * N0 * (ps.energy / u.GeV)**(-alpha) /u.GeV * losses *
                                ps.energy** 2.75 * c / (4 * np.pi * u.sr)).to(u.GeV ** (7 / 4) / (u.cm ** 2 * u.s * u.sr))


                            figures[-1], ax  = proton_spectrum(
                                                    pdf=ps, 
                                                    flux=flux,
                                                    fig=figures[-1], ax=ax, label=f'$\delta = {delta}$',
                                                    ls='-')
                            ax.legend(loc='lower left')

                    ax.text(0.05, 0.95, f'{acc_cfg["name"]} ({t:.0f})',
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)

                    ax.text(0.05, 0.88, f'$\chi$ = {chi}, $n_0$={n0}, Ebudget={eta*ps.E_SN:.1e}',
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)


    with PdfPages(output+f'ProtonSpectra_{acc.name}.pdf') as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')


if __name__ == '__main__':
    main()
