import astropy.units as u
import numpy as np
from astropy.io import fits

import click
from astropy.io.misc import yaml

@click.command()
@click.argument('pdf_cube_path')
@click.argument('model')
@click.option('--model_config_path', 
                default='hadronic_models.yaml',
                type=click.Path(exists=True, dir_okay=False))


def main(pdf_cube_path, model, model_config_path):

    with open(model_config_path) as file:
        model_cfg = yaml.load(file)[model]

    file = fits.open(pdf_cube_path)
    header = file[0].header
    pdf = (file[0].data) * u.Unit(header['BUNIT'])
    
    energies = np.logspace(header['CRVAL3'],
                     header['CRVAL3'] + (header['NAXIS3'] - 1) * header['CDELT3'],
                     header['NAXIS3'])* u.Unit(header['CUNIT3'])
    
    etas = model_cfg['etas']
    alpha = model_cfg['alpha']

    for eta in etas:
# Will loop over alpha later
#        for alpha in alphas:

        Ebudget = eta * header['HESN_ERG']*u.erg
        Emin = 1*u.GeV
        Emax = header['HEMT_TEV']*u.TeV
        N0 = (Ebudget * (1-alpha) / (((Emax/u.GeV)**(1-alpha) - (Emin/u.GeV)**(1-alpha))*u.GeV)).decompose()

        cool_time = (6e7 * (header['HN_CM3']) ** (-1) * u.yr)
        losses = np.exp(- ((alpha - 1) * header['HT_KYR']*u.kyr  / cool_time)) 

        Esize, xsize, ysize = pdf.shape
        Jp = np.empty((Esize, xsize, ysize))
        for i in range(len(energies)):
            Jp[i, :,:] = (pdf[i,:,:] * losses * N0 * (energies[i] / u.GeV)**(-alpha) /u.GeV).to(u.cm**(-3)/u.GeV)

        Jp = Jp *  u.cm**(-3)/u.GeV

        basepdf = pdf_cube_path.split('HadronicPDF_')[1]
        filename = f'HadronicProtonFlux_Ebudget{Ebudget.to(u.erg).value:.1e}_{basepdf}'

        step = np.diff(np.log10(energies[:].to(u.TeV).value))[0]

        E0 = np.log10(energies[0].to(u.TeV).value)

        hdr = fits.PrimaryHDU(data=Jp.value)
        hdr.header['BUNIT'] = f'{Jp.unit}'

        hdr.header['CTYPE1'] = header['CTYPE1']
        hdr.header['CRVAL1'] = header['CRVAL1']
        hdr.header['CRPIX1'] = header['CRPIX1']
        hdr.header['CDELT1'] = header['CDELT1']

        hdr.header['CTYPE2'] = header['CTYPE2']
        hdr.header['CRVAL2'] = header['CRVAL2']
        hdr.header['CRPIX2'] = header['CRPIX2']
        hdr.header['CDELT2'] = header['CDELT2']

        hdr.header['CTYPE3'] = 'log ENERGY'
        hdr.header['CUNIT3'] = 'TeV'
        hdr.header['CRVAL3'] = E0
        hdr.header['CRPIX3'] = 1.
        hdr.header['CDELT3'] = step

        hdr.header['HNAME'] = header['HNAME']
        hdr.header['HT_KYR'] = header['HT_KYR']
        hdr.header['HCHI'] = header['HCHI']
        hdr.header['HDELTA'] = header['HDELTA']
        hdr.header['HD0_CM2S'] = header['HD0_CM2S']
        hdr.header['HN_CM3'] = header['HN_CM3']
        hdr.header['HN0_CM3'] = header['HN0_CM3']
        hdr.header['HESN_ERG'] = header['HESN_ERG']
        hdr.header['HEM100'] = header['HEM100']
        hdr.header['HEMT_TEV'] = header['HEMT_TEV']
        hdr.header['HTST_KYR'] = header['HTST_KYR']
        hdr.header['HDELTAP'] = header['HDELTAP']
        hdr.header['HEBUDGET_ERG'] = Ebudget.to(u.erg).value

        hdr.header['AUTHOR'] = header['AUTHOR']

        hdr.writeto(filename, overwrite=True)



if __name__ == '__main__':
    main()
