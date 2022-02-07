from astropy import units as u
import numpy as np

class GammaSource:
    """
    A class used to define a gamma-ray source
    """
    @u.quantity_input(glon=u.deg, glat=u.deg, radius=u.deg,
                      flux_GeV=1 / (u.s * u.cm ** 2),
                      flux_E_GeV=u.TeV,
                      flux_TeV=1 / (u.s * u.cm ** 2),
                      flux_E_TeV=u.TeV)
    def __init__(self, name, spectral_index, glon, glat, radius, 
                  flux_GeV, flux_E_GeV, flux_TeV, flux_E_TeV):
        """
        Parameters
        ----------------------
        name : str
            The name of the gamma-ray source
        spectral_index : float
            The spectral index of the gamma-ray source in the TeV regime
        glon : float
            The galactic longitude of the gamma-ray source
        glat : float
            The galactic latitude of the gamma-ray source
        radius : float
            The radius of the gamma-ray source
        flux_GeV : float
            The integral photon flux above a certain energy for the GeV gamma-ray source
        flux_E_GeV : float
            The lower energy limit for the integral photon flux at GeV energies
        flux_TeV : float
            The integral photon flux above a certain energy for the TeV gamma-ray source
        flux_E_TeV : float
            The lower energy limit for the integral photon flux at TeV energies


        Examples
        ----------------------
        >>> hess = GammaSource(name='HESS J1804', spectral_index=2.69,
           flux_GeV=1.56e-9 / (u.s * u.cm**2),
           flux_E_GeV=0.01 * u.GeV,
           flux_TeV=5.3e-11/(u.s * u.cm**2),
           flux_E_TeV=0.2 * u.TeV)

        """
        self.name = name
        self.spectral_index = spectral_index
        self.glon = glon
        self.glat = glat
        self.radius = radius
        self.flux_GeV = flux_GeV
        self.flux_E_GeV = flux_E_GeV
        self.flux_TeV = flux_TeV
        self.flux_E_TeV = flux_E_TeV


class Accelerator:
    """
    A class used to define the supernova remnant

    """

    @u.quantity_input(glon=u.deg, glat=u.deg, radius=u.deg,
                      age=u.kyr, distance=u.kpc,
                      gas_mass=u.solMass)
    def __init__(self, name, glon, glat, radius, age, distance, gas_mass):
        """
        Parameters
        ----------------------
        name : str
            The name of the supernova remnant
        glon : float
            The galactic longitude of the accelerator
        glat :float
            The galactic latitude of the accelerator
        radius : float
            The radius of the accelerator
        age : float
            The age of the accelerator
        distance: float
            The distance from Earth to the accelerator
        gas_mass : float
            The solar mass of the interstellar medium surrounding the accelerator (at the distance to the accelerator)

        Examples
        ----------------------
        >>> G8 = Accelerator(name='SNR G8.7-0.1',
                 			 glon=8.744205 * u.deg, glat=-0.096288 * u.deg, radius=0.375*u.deg,
                 			 age=15 * u.kyr, distance=4.5 * u.kpc, gas_mass = 7.9e5 * u.solMass)
        """
        self.name = name
        self.glon = glon
        self.glat = glat
        self.radius = radius
        self.age = age
        self.distance = distance
        self.gas_mass = gas_mass
        
        self.radius_pc = self.distance * np.tan(self.radius)
        
