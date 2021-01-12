import numpy as np
from astropy import units as u
from astropy.constants import c
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from matplotlib.patches import Circle
from astropy import coordinates as coord


def add_SNR(snr, ax, **kwargs):
    """
    Plot the supernova remnant counterpart

    Parameters
    ----------
    snr : class
        Definition for the counterpart supernova remnant
    ax : str
        Set the axis for the figure
    kwargs : keywords
        Initialisation keywords

    Examples
    ----------------------
    >>> add_SNR(G8, ax, facecolor='None', edgecolor='white', ls='--', lw=2)
    """
    font1 = {'family': 'sans-serif', 'weight': 'bold', 'size': 13}
    snr_region = Circle((snr.glon.value, snr.glat.value), snr.radius.to(u.deg).value,
                        transform=ax.get_transform('galactic'), **kwargs)
    ax.add_patch(snr_region)

    ax.text(snr.glon.value, snr.glat.value + (snr.radius * 1.1).to(u.deg).value, snr.name,
            color=snr_region.get_edgecolor(),
            transform=ax.get_transform('galactic'),
            fontdict=font1, zorder=10,
            horizontalalignment='center')
    return ax


def add_SNR_centre(snr, ax, **kwargs):
    """
    Plot the supernova remnant counterpart

    Parameters
    ----------
    snr : class
        Definition for the counterpart supernova remnant
    ax : str
        Set the axis for the figure
    kwargs : keywords
        Initialisation keywords

    Examples
    ----------------------
    >>> add_SNR_centre(PJ, ax, facecolor='black', ls='-', lw=2)
    """
    font1 = {'family': 'sans-serif', 'weight': 'bold', 'size': 13}
    snr_region = Circle((snr.glon.value, snr.glat.value), 0.02,
                        transform=ax.get_transform('galactic'), **kwargs)
    ax.add_patch(snr_region)

    ax.text(snr.glon.value, snr.glat.value - (snr.radius * 0.4).to(u.deg).value, snr.name,
            color=snr_region.get_facecolor(),
            transform=ax.get_transform('galactic'),
            fontdict=font1, zorder=10,
            horizontalalignment='center')
    return ax


def add_gamma_src(src_fits, con_levels, ax, **kwargs):
    """
    Plot the contours from your gamma-ray source

    Parameters
    ----------
    src_fits : str
        The gamma-ray map you wish to obtain contours from
    con_levels : list
        Contours levels for the fits file
    ax : str
        Set the axis for the figure
    kwargs : keywords
        Initialisation keywords

    Examples
    ----------------------
    >>> add_gamma_src('HESSJ1804.fits', [10,20,30,40], ax, colors='black')
    """
    src = fits.open(src_fits)
    con_data = src[1].data
    con_header = src[1].header
    con_header.remove('CUNIT1')
    con_header.remove('CUNIT2')

    ax.contour(con_data, transform=ax.get_transform(WCS(con_header)),
               levels=con_levels, **kwargs)

    return ax


def set_limits(ax, header, l=[7.5, 9.5], b=[-0.5, 0.5]):
    """
    Set the limits for the galactic plots

    Parameters
    ----------
    ax : str
        Set the axis for the figure
    header : str
        The header of the fits file
    l : array, optional
        The galactic longitude to plot (default is l=[7.5, 9.5])
    b : array, optional
        The galactic latitude to plot (default is b=[-0.5, 0.5])
    """
    c1 = coord.SkyCoord(l=l[0] * u.deg, b=b[0] * u.deg, frame='galactic').to_pixel(WCS(header))
    c2 = coord.SkyCoord(l=l[1] * u.deg, b=b[1] * u.deg, frame='galactic').to_pixel(WCS(header))
    ax.set_xlim(c2[0], c1[0])
    ax.set_ylim(c2[1], c1[1])
    ax.invert_yaxis()

    return ax


def proton_spectrum(pdf, flux, label=None, age_label=False, dif_label=False, initial_label=False,
                    fig=None, ax=None, **kwargs):
    """
    Plot the proton spectrum

    Parameters
    ----------
    pdf : array
        The pdf you want to plot
    flux : array
        The flux you want to plot
    age_label : bool
        Plot the age label (default is False)
    dif_label : bool
        Plot the diffusion label (default is False)
    initial_label : bool
        Plot the initial spectral parameters (default is False)
    fig : str
        Set the figure for the plot
    ax : str
        Set the axis for the figure
    **kwargs : keywords
        Initialisation keywords

    Examples
    ----------------------
    >>> Ep = np.logspace(0, 6, 50)*u.GeV
    >>> ps.calc_pdf(E=Ep, R=[30]*u.pc, t=t)
    >>> fig, ax  = proton_spectrum(pdf=ps, flux=flux, diff_label=True)

    """
    if fig is None:
        fig = plt.figure(figsize=(8, 6))

    if ax is None:
        ax = plt.gca()

    if label is None:
        label = f'{pdf.t:1.0f}'
    ax.plot(pdf.energy, (flux.to(u.GeV ** (7 / 4) / (u.cm ** 2 * u.s * u.sr))).value,
            label=label,
            **kwargs)

    if dif_label:
        ax.text(0.03, 0.93, f'$\chi$={pdf.dif_chi},   $\delta$={pdf.dif_delta}',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontsize=15, color='black')
        
    if initial_label:
        ax.text(0.03, 0.83, f'$\delta_p$={pdf.deltap:.2f}, $n_0$={pdf.n0.value}'+
                     ' cm$^{-3}$, E$_{esc}$='+f'{pdf.Eesc:.1f}',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontsize=15, color='black')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylim(1e-10, 1e5)

    ax.set_xlabel('Energy (GeV)')
    ax.set_ylabel('JE$^{2.75}$ (GeV$^{1.75}$cm$^{-2}$s$^{-1}$ sr$^{-1}$)')

    if age_label:
        ax.legend(title='Age of SNR',
                  loc='lower right',
                  ncol=1,
                  labelspacing=0.3)


    return fig, ax






def add_kcr(ax=None, label='CR Flux at Earth', **kwargs):
    """
    Plot the solar neighbour CR flux value

    Parameters
    ----------
    ax : str
        Set the axis for the figure
    label : bool, optional
        Set the label of the line (default is 'CR Flux at Earth')
    **kwargs : keywords
        Initialisation keywords
    """

    line = ax.axhline(2.2, **kwargs)

    ax.text(0.9e5, 2.2 - 0.3, label,
            fontsize=14, color=line.get_color(),
            horizontalalignment='right',
            verticalalignment='top')

    return ax


@u.quantity_input(emin=u.GeV, emax=u.GeV)
def add_kcr_src(gamma_src, snr, emin, emax, ax=None, label='', **kwargs):
    """
    Plot the cosmic ray enhancement factors for GeV and TeV energies

    Parameters
    ----------
    gamma_src: class
        The gamma-ray source class
    snr: class
        Definition for the counterpart supernova remnant
    emin : float
        Minimum energy from gamma-ray detection in GeV
    emax : float
        Maximum energy from the gamma-ray detection in GeV
    ax : str
        Set the axis for the figure
    label : bool
        Set the label of the kCR line
    **kwargs : keywords
        Initialisation keywords
    """

    if np.mean([emin.to(u.GeV).value, emax.to(u.GeV).value]) > 1000:
        kcr = ((gamma_src.flux_TeV / (2.85e-13 / (u.cm ** 2 * u.s))) *
               (gamma_src.flux_E_TeV / u.TeV) ** 1.6 *
               ((snr.distance / u.kpc) ** 2 / ((snr.gas_mass.to(u.solMass)) / (1e5 * u.solMass))))
    else:
        kcr = ((gamma_src.flux_GeV / (1.45e-13 / (u.cm ** 2 * u.s))) *
               (gamma_src.flux_E_GeV / u.TeV) ** 1.75 *
               ((snr.distance / u.kpc) ** 2 / ((snr.gas_mass.to(u.solMass)) / (1e5 * u.solMass))))

    line, = plt.plot([emin.to(u.GeV).value, emax.to(u.GeV).value],
                     [kcr * 2.2, kcr * 2.2], **kwargs)

    ax.text(10 ** np.mean([np.log10(emin.to(u.GeV).value), np.log10(emax.to(u.GeV).value)]),
            kcr,
            label, fontsize=14, color=line.get_color(),
            horizontalalignment='center',
            verticalalignment='baseline')
    return ax




