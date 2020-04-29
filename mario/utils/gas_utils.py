import numpy as np
from astropy import units as u
from spectral_cube import SpectralCube
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


class TotalColumnDensity:
    """
    A class to calculate the total column density for a given CO and HI cube

    """

    @u.quantity_input(v1=u.km / u.s, v2=u.km / u.s,
                      X_CO=u.s * 1 / (u.K * u.km * u.cm ** 2),
                      X_HI=u.s * 1 / (u.K * u.km * u.cm ** 2))
    def __init__(self, HI_path, CO_path, v1, v2,
                 X_CO=1.8e20 * u.s * 1 / (u.K * u.km * u.cm ** 2),
                 X_HI=1.823e18 * u.s * 1 / (u.K * u.km * u.cm ** 2)):
        """
        Parameters
        ----------------------
        HI_path : str
            Path to the gas map for the HI
        CO_path : str
            Path to the gas map for the CO
        v1 : float
            The first velocity for integration
        v2 : float
            The second velocity for integration
        X_CO : float
            The X-factor for CO
        X_HI : float
            The X-factor for HI
        Examples
        ----------------------
        >>> HI = 'Documents/Fits_Files/HI_cube.fits'
        >>> CO = 'Documents/Fits_Files/12CO_cube.fits'
        >>> TCD = TotalColumnDensity(HI, CO, 26*u.km/u.s, 56*u.km/u.s)
        >>> total = TCD.calc_total_col()

        """

        self.HI_path = HI_path
        self.HI_cube = SpectralCube.read(self.HI_path)

        self.CO_path = CO_path
        self.CO_cube = SpectralCube.read(self.CO_path)

        self.v1 = v1
        self.v2 = v2

        self.X_CO = X_CO
        self.X_HI = X_HI

    def calc_total_col(self):
        """
        Calculate the total column density for a given CO and HI cube

        """
        # Calculation of moment 0 arrays
        HI_cube_vel_range = self.HI_cube.spectral_slab(self.v1, self.v2)
        HI_mom0 = HI_cube_vel_range.with_spectral_unit(u.km / u.s).moment(order=0)

        CO_cube_vel_range = self.CO_cube.spectral_slab(self.v1, self.v2)
        CO_mom0 = CO_cube_vel_range.with_spectral_unit(u.km / u.s).moment(order=0)

        # Calculation of total column density
        HI_col_dens = HI_mom0 * self.X_HI
        CO_col_dens = CO_mom0 * self.X_CO
        total_col_dens = 2 * CO_col_dens + HI_col_dens

        return total_col_dens

    def total_header(self):
        """
        Access to the header from the total column density map

        """
        total_header = self.calc_total_col().header
        return total_header

    def save_fits(self, author='Kirsty Feijen', filename='total_column.fits'):
        """
        Save the total column density to a fits file

        Parameters
        ----------------------
        author : str, optional
            Author of the fits file (default 'Kirsty Feijen')
        filename : str, optional
            The name to save your fits to (default 'total_column.fits')

        """
        hdu = fits.PrimaryHDU(data=self.calc_total_col().value)
        hdu.header = self.total_header()
        hdu.header['AUTHOR'] = author
        hdu.writeto(filename, overwrite=True)

        return


class GasParameters:
    """
    A class to calculate the gas parameters: number density and distance to cells

    """

    def __init__(self, snr, gas_map_path):
        """
        Parameters
        ----------------------
        snr : class
            The class for the supernova remnant counterpart
        gas_map_path : str
            Path to the gas map in the format of fits

        Examples
        ----------------------
        >>> gas_map_path = 'Total_26_56.fits'
        >>> gp = GasParameters(snr=G8, gas_map_path=gas_map_path)

        """
        self.snr = snr

        file = fits.open(gas_map_path)
        self.total_col_dens = file[0].data * u.cm ** (-2)

        if len(np.shape(self.total_col_dens)) == 3:
            self.total_col_dens = file[0].data[0][:][:] * u.cm ** (-2)

        self.header = file[0].header

        # Width of pixel
        self.theta_pixel = (self.header['CDELT2']) * u.deg

    def nH2_cell(self):
        """
        Calculates the number density of each pixel in your map

        Returns
        ----------------------
        array
            The number density of each pixel

        """
        # Calculation of box depth
        depth = self.snr.distance.to(u.cm) * np.tan(0.45 * u.deg)

        # Calculation of volume
        area = (self.snr.distance.to(u.cm) * np.tan(self.theta_pixel)) ** 2
        volume = depth * area

        # Calculation of number density
        nH2 = (self.total_col_dens / volume) * area
        nH2[nH2 < 0] = 0
        nH2[np.isnan(nH2)] = 0

        return nH2

    def dist_cell(self):
        """
        Calculates the distance to each pixel in your map, taking the coordinates
        of the counterpart as the centre.

        Returns
        ----------------------
        array
            The distance to each pixel from the accelerator
        """
        # Set the coordinates of the counterpart
        OBJ_coord = SkyCoord(self.snr.glon, self.snr.glat, frame='galactic')

        # Converting from pixel coords to sky coords
        xvalue, yvalue = self.total_col_dens.shape
        xx, yy = np.meshgrid(range(yvalue), range(xvalue))
        wx, wy = WCS(self.header).wcs_pix2world(xx, yy, 1)
        pixel_coord = SkyCoord(wx * u.deg, wy * u.deg, frame='galactic')

        # Calculate the distance to each cell
        theta_sep = (OBJ_coord.separation(pixel_coord)).deg * u.deg
        dist = self.snr.distance.to(u.cm) * np.tan(theta_sep)

        return dist
