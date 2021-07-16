"""
Script used to create yaml file for hadronic scenario model.
"""

from astropy import units as u
import numpy as np 

from mario.sources import Accelerator
G8 = Accelerator(name='SNR G8.7-0.1', 
         glon=8.744205 * u.deg, glat=-0.096288 * u.deg, radius=0.375*u.deg,
         age=15 * u.kyr, distance=4.5 * u.kpc, gas_mass=7.9e5 * u.solMass)
         
E_SN = [1e51, 1e52]*u.erg

from mario.gas_utilities import calc_n0

n0_G8_15 = calc_n0(G8.distance * np.tan(G8.radius), G8.age, E_SN)

import yaml
s1_G8_15 = """s1_G8_15:
  dif_chi:
    - 0.1
    - 0.01
    - 0.001

  dif_delta:
    - 0.3
    - 0.4
    - 0.5
    - 0.6
    - 0.7
  
  deltaps:
    - 0.2
    - 1.4
    - 2.5

  Emax_t_ST0s: 
    - !astropy.units.Quantity
      unit: &ide1 !astropy.units.Unit {unit: PeV}
      value: 1
    - !astropy.units.Quantity
      unit: *ide1
      value: 5
      
  Mejs: 
    - !astropy.units.Quantity
      unit: &idm1 !astropy.units.Unit {unit: solMass}
      value: 1
    - !astropy.units.Quantity
      unit: *idm1
      value: 10
"""
with open('hadronic_models.yaml', 'w') as f:
    f.write(s1_G8_15)
    
n0s_0 = """
  n0s:
    - !astropy.units.Quantity
      unit: &id001 !astropy.units.Unit {unit: cm**(-3)}
      value: %.3f

  E_SN: !astropy.units.Quantity
    value: %s
    unit: !astropy.units.Unit {unit: erg}
      
      """ %(n0_G8_15[0].value, E_SN[0].value)

with open('hadronic_models.yaml', 'a') as f:
    f.write(n0s_0)
    


s2_G8_15 = """
s2_G8_15:
  dif_chi:
    - 0.1
    - 0.01
    - 0.001

  dif_delta:
    - 0.3
    - 0.4
    - 0.5
    - 0.6
    - 0.7

  deltaps:
    - 0.2
    - 1.4
    - 2.5

  Emax_t_ST0s: 
    - !astropy.units.Quantity
      unit: &ide2 !astropy.units.Unit {unit: PeV}
      value: 1
    - !astropy.units.Quantity
      unit: *ide2
      value: 5

  Mejs: 
    - !astropy.units.Quantity
      unit: &idm2 !astropy.units.Unit {unit: solMass}
      value: 10
    - !astropy.units.Quantity
      unit: *idm2
      value: 20
"""
with open('hadronic_models.yaml', 'a') as f:
    f.write(s2_G8_15)
    
n0s_1 = """
  n0s:
    - !astropy.units.Quantity
      unit: &id002 !astropy.units.Unit {unit: cm**(-3)}
      value: %.2f

  E_SN: !astropy.units.Quantity
    value: %s
    unit: !astropy.units.Unit {unit: erg}
      
      """ %(n0_G8_15[1].value, E_SN[1].value)

with open('hadronic_models.yaml', 'a') as f:
    f.write(n0s_1)
