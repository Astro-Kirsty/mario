import numpy as np
from astropy import units as u
from astropy.constants import c
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib as mpl
from astropy.wcs import WCS
from astropy.io import fits
from matplotlib.patches import Circle
from spectral_cube import SpectralCube


def plot_SNR(snr, ax, text_lon, text_lat, color='blue', ls='--', lw=2):
    """
    Plot the supernova remnant counterpart

    Parameters
    ----------
    snr : class
        Definition for the counterpart supernova remnant
    ax : str
        Set the axis for the figure
    text_lon : float
        Set the galactic longitude of the label
    text_lat : float
        Set the galactic latitude of the label
    color : str
        Color name or rgb code
    ls : str, default '-.'
        Linstyle
    lw : float, default 2
        Linewidth
    Examples
    ----------------------
    >>> plot_SNR(G8, ax, text_lon=9.02, text_lat=0.3)
    """
    font1 = {'family': 'sans-serif', 'color': 'darkred', 'weight': 'bold', 'size': 13}
    c1 = Circle((snr.glon.value, snr.glat.value), snr.radius.to(u.deg).value,
                transform=ax.get_transform('galactic'),
                edgecolor=color, ls=ls, lw=lw, facecolor='None')
    ax.add_patch(c1)

    ax.text(text_lon, text_lat, snr.name, color=color,
            transform=ax.get_transform('galactic'),
            fontdict=font1, zorder=10)
    return ax


def plot_source(src_fits, con_levels, ax, colors):
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
    colors : str
        Color name or rgb code

    Examples
    ----------------------
    >>> plot_source('HESSJ1804.fits', [10,20,30,40], ax, 'black')
    """
    src = fits.open(src_fits)
    con_data = src[1].data
    con_header = src[1].header
    con_header.remove('CUNIT1')
    con_header.remove('CUNIT2')

    ax.contour(con_data, transform=ax.get_transform(WCS(con_header)),
               levels=con_levels, colors=colors)

    return ax


def plot_proton_spectrum(spectrum, age_label=True, diff_label=True, diff_fontsize=15, diff_color='black',
                         fig=None, ax=None, color='#00CDCD'):
    """
    Plot the proton spectrum

    Parameters
    ----------
    spectrum : array
        The proton spectra you want to plot
    age_label : bool
        Plot the age label (default True)
    diff_label : bool
        Plot the diffusion label (default True)
    diff_fontsize :  float
        Fontsize of the diffusion label (default 15)
    diff_color : float
        Font color of the diffusion label (default 'black')
    fig : str
        Set the figure for the plot
    ax : str
        Set the axis for the figure
    color : str, default 'cyan'
        Color name or rgb code

    Examples
    ----------------------
    >>> from plotting.plotting import plot_proton_spectrum, add_kcr_src, add_kcr
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure(figsize=(6, 4))
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> G8.age = 15*u.kyr
    >>> ps.proton_flux(Ep, rcloud=ps.dist_cloud)
    >>> fig, ax  = plot_proton_spectrum(spectrum=ps, fig=fig, ax=ax, diff_label=False)

    """
    if fig is None:
        fig = plt.figure(figsize=(8, 6))

    if ax is None:
        ax = plt.gca()

    energy = spectrum.energies[:, 0, 0]
    flux = (spectrum.fluxes[:, 0, 0] *
            energy ** 2.75 * c / (4 * np.pi * u.sr))

    ax.plot(energy, np.log10((flux.to(u.GeV ** (7 / 4) / (u.cm ** 2 * u.s * u.sr))).value),
            color=color,
            label='%.f kyrs' % spectrum.snr.age.value)

    if diff_label:
        ax.text(0.03, 0.93, '$\delta$=' f'{spectrum.diff_delta}' ',    $\chi$=' f'{spectrum.diff_chi}',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontsize=diff_fontsize, color=diff_color)

    ax.set_xscale('log')
    ax.set_xlim(7e1, 1e5)
    ax.set_ylim(-3, 4)

    ax.set_xlabel('Energy (GeV)')
    ax.set_ylabel('log(JE$^{2.75}$) (GeV$^{1.75}$cm$^{-2}$s$^{-1}$ sr$^{-1}$)')

    if age_label:
        ax.legend(title='Age of SNR',
                  loc='lower right',
                  ncol=1,
                  labelspacing=0.3)
    return fig, ax


def add_kcr(ax=None, label='CR Flux at Earth', fontsize=14, color='magenta', ls='-.', lw=2):
    """
    Plot the solar neighbour CR flux value

    Parameters
    ----------
    ax : str
        Set the axis for the figure
    label : bool
        Set the label of the line (default 'CR Flux at Earth')
    fontsize : float
        Fontsize of the label (default 14)
    color : str
        Color name or rgb code (default 'magenta')
    ls : str
        Linstyle (default '-.')
    lw : float
        Linewidth (default 2)
    """

    ax.axhline(np.log10(2.2),
               color=color,
               lw=lw, ls=ls)

    ax.text(0.9e5, np.log10(2), label,
            color=color, fontsize=fontsize,
            horizontalalignment='right',
            verticalalignment='top')

    return ax


@u.quantity_input(emin=u.GeV, emax=u.GeV)
def add_kcr_src(spectrum, emin, emax, ax=None, label='', fontsize=14, color='green', ls='--', lw=2):
    """
    Plot the cosmic ray enhancement factors for GeV and TeV energies

    Parameters
    ----------
    spectrum : array
        The proton spectra you want to plot
    emin : float
        Minimum energy from gamma-ray detection in GeV
    emax : float
        Maximum energy from the gamma-ray detection in GeV
    ax : str
        Set the axis for the figure
    label : bool
        Set the label of the kCR line
    fontsize : float
        Fontsize of the label (default 14)
    color : bool, default 'green'
        Color name or rgb code
    ls : str, default '--'
        Linstyle
    lw : float, default 2
        Linewidth
    """

    if np.mean([emin.to(u.GeV).value, emax.to(u.GeV).value]) > 1000:
        k = ((spectrum.gamma_src.flux_TeV / (2.85e-13 / (u.cm ** 2 * u.s))) *
             (spectrum.gamma_src.flux_E_TeV / u.TeV) ** 1.6 *
             ((spectrum.snr.distance / u.kpc) ** 2 / (spectrum.gamma_src.mass / (1e5 * u.solMass))))
    else:
        k = ((spectrum.gamma_src.flux_GeV / (1.45e-13 / (u.cm ** 2 * u.s))) *
             (spectrum.gamma_src.flux_E_GeV / u.TeV) ** 1.75 *
             ((spectrum.snr.distance / u.kpc) ** 2 / (spectrum.gamma_src.mass / (1e5 * u.solMass))))

    ax.hlines(np.log10(k * 2.2),
              color=color, lw=lw, ls=ls,
              xmin=emin.to(u.GeV).value, xmax=emax.to(u.GeV).value, zorder=10)

    ax.text(10 ** np.mean([np.log10(emin.to(u.GeV).value), np.log10(emax.to(u.GeV).value)]), np.log10(k * 2.2), label,
            color=color, fontsize=fontsize,
            horizontalalignment='center',
            verticalalignment='top')

    return ax


def plot_gamma_flux(cube_path, cb_label='Gamma-ray flux (TeV s$^{-1}$ cm$^{-2}$)', fontsize=14,
                    zaxis=0, fig=None, cmap=plt.cm.hot_r):
    """
    Plot the predicted gamma-ray flux

    Parameters
    ----------
    cube_path : str
        The path to the cube you want to plot
    cb_label : str
        The label of the colorbar
    fontsize : float
        Fontsize of the label (default 14)
    zaxis : int
        The energy to access from the zaxis
    fig : str
        Set the figure for the plot
    cmap : str
        Choose the colormap for your plot


    Examples
    ----------------------
    >>> fig, ax = plot_gamma_flux('Gflux_cube.fits', zaxis=1)

    """
    cube = SpectralCube.read(cube_path)
    header = cube.header
    header.remove('CRPIX3')
    header.remove('CDELT3')
    header.remove('NAXIS3')
    header.remove('CRVAL3')
    header.remove('CTYPE3')
    header.remove('CUNIT3')
    header['NAXIS'] = 2
    header['WCSAXES'] = 2
    wcs = WCS(header)

    if fig is None:
        fig = plt.figure(figsize=(15, 4))

    ax = fig.add_subplot(projection=wcs)

    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    lon = ax.coords['glon']
    lat = ax.coords['glat']
    lon.set_axislabel('Galactic Longitude')
    lat.set_axislabel('Galactic Latitude')

    lat.set_major_formatter('d.dd')
    lon.set_major_formatter('d.dd')
    lon.set_ticklabel_position('b')

    lat.set_ticklabel(rotation=90)
    lat.set_ticks_position('l')
    lat.set_ticklabel_position('l')
    lat.set_axislabel_position('l')

    lon.set_ticks(spacing=0.4 * u.deg, color='black')
    lat.set_ticks(spacing=0.4 * u.deg, color='black')
    lon.display_minor_ticks(True)
    lon.set_minor_frequency(4)
    lat.display_minor_ticks(True)
    lat.set_minor_frequency(4)

    img = ax.imshow(cube[zaxis, :, :].value, cmap=cmap)

    ax.text(0.02, 0.9, 'E$_\gamma$=%.1f TeV' % 10 ** cube.spectral_axis[zaxis].value,
            transform=ax.transAxes, color='black', fontsize=fontsize)

    cb = fig.colorbar(img, pad=0.005)
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()
    cb.set_label(cb_label, rotation=270, labelpad=12)

    zlim, ylim, xlim = np.shape(cube)
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, ylim])

    return fig, ax


def plot_gamma_counts(counts_path, fig=None, cmap=plt.cm.hot_r, counts_label='HESS counts', fontsize=14):
    """
    Plot the simulated gamma-ray counts

    Parameters
    ----------
    counts_path : str
        The path to the counts map you want to plot
    fig : str
        Set the figure for the plot
    cmap : str
        Choose the colormap for the plot
    counts_label
        The title of the counts map
    fontsize : float
        Fontsize of the label (default 14)

    Examples
    ----------------------
    >>> fig, ax = plot_gamma_counts('HESScounts.fits')

    """
    file = fits.open(counts_path)
    counts_data = file[0].data
    header = file[0].header
    wcs = WCS(header)

    if fig is None:
        fig = plt.figure(figsize=(15, 4))

    ax = fig.add_subplot(projection=wcs)

    lon = ax.coords['glon']
    lat = ax.coords['glat']
    lon.set_axislabel('Galactic Longitude')
    lat.set_axislabel('Galactic Latitude')

    lat.set_major_formatter('d.dd')
    lon.set_major_formatter('d.dd')
    lon.set_ticklabel_position('b')

    lat.set_ticklabel(rotation=90)
    lat.set_ticks_position('l')
    lat.set_ticklabel_position('l')
    lat.set_axislabel_position('l')

    lon.set_ticks(spacing=0.4 * u.deg, color='black')
    lat.set_ticks(spacing=0.4 * u.deg, color='black')
    lon.display_minor_ticks(True)
    lon.set_minor_frequency(4)
    lat.display_minor_ticks(True)
    lat.set_minor_frequency(4)

    img = ax.imshow(counts_data, cmap=cmap)

    cb = fig.colorbar(img, pad=0.005)
    cb.set_label('Counts', rotation=270, labelpad=12)

    ax.text(0.02, 0.9, counts_label, transform=ax.transAxes,
            fontweight='bold', color='black', fontsize=fontsize)

    ylim, xlim = counts_data.shape
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, ylim])

    return fig, ax


