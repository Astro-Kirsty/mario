from astropy import units as u


class Source:
    """
    A class used to define a gamma-ray source
    """

    @u.quantity_input(flux_GeV=1 / (u.s * u.cm ** 2),
                      flux_E_GeV=u.TeV,
                      flux_TeV=1 / (u.s * u.cm ** 2),
                      flux_E_TeV=u.TeV,
                      mass=u.solMass)
    def __init__(self, name, spectral_index, flux_GeV, flux_E_GeV, flux_TeV, flux_E_TeV, mass):
        """
        Parameters
        ----------------------
        name : str
            The name of the gamma-ray source
        spectral_index : float
            The spectral index of the gamma-ray source
        flux_GeV : float
            The photon flux above a certain energy of the GeV gamma-ray source
        flux_E_GeV : float
            The energy of the photon flux
        flux_TeV : float
            The photon flux above a certain energy of the TeV gamma-ray source
        flux_E_TeV : float
            The energy of the photon flux
        mass : float
            The solar mass of the interstellar medium surrounding the gamma-ray source

        Returns
        ----------------------
        dict
            Description of the gamma-ray source.

        Examples
        ----------------------
        >>> hess = Source(name='HESS J1806', spectral_index=2.69,
           flux_GeV=1.56e-9 / (u.s * u.cm**2),
           flux_E_GeV=0.01 * u.GeV,
           flux_TeV=5.3e-11/(u.s * u.cm**2),
           flux_E_TeV=0.2 * u.TeV,
           mass=7.9e5 * u.solMass)

        """
        self.name = name
        self.spectral_index = spectral_index
        self.flux_GeV = flux_GeV
        self.flux_E_GeV = flux_E_GeV
        self.flux_TeV = flux_TeV
        self.flux_E_TeV = flux_E_TeV
        self.mass = mass


class SNR:
    """
    A class used to define the counterpart supernova remnant

    """

    @u.quantity_input(glon=u.deg, glat=u.deg, radius=u.deg,
                      age=u.kyr, distance=u.kpc)
    def __init__(self, name, glon, glat, radius, age, distance):
        """
        Parameters
        ----------------------
        name : str
            The name of the supernova remnant counterpart
        glon : float
            The galactic longitude of the supernova remnant counterpart
        glat :float
            The galactic latitude of the supernova remnant counterpart
        radius : float
            The radius of the supernova remnant counterpart
        age : float
            The age of the supernova remnant counterpart
        distance: float
            The distance from Earth to the supernova remnant counterpart

        Returns
        ----------------------
        dict
            Description of the supernova remnant counterpart

        Examples
        ----------------------
        >>> G8 = SNR(name='SNR G8.7-0.1',
                 glon=8.744205 * u.deg, glat=-0.096288 * u.deg, radius=0.375*u.deg,
                 age=15 * u.kyr, distance=4.5 * u.kpc)
        """
        self.name = name
        self.glon = glon
        self.glat = glat
        self.radius = radius
        self.age = age
        self.distance = distance
