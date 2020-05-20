# Mario - Modelling gAlactic gamma-Ray emissIOn

![](images/teaser.png)

The aim of `mario` is to provide users with proton spectra, gamma-ray flux maps and simulated excess counts maps for a given gamma-ray source. It is designed for investigating the potential counterparts of high energy gamma-ray sources, as detected through various telescopes including, but not limited to the High Energy Stereoscopic System and Fermi-LAT. 

## Documentation 
The code, and examples are hosted at https://github.com/Astro-Kirsty/mario


## Installation
The most stable version of `mario` can be easily installed from [PyPI]()
with [pip](https://pip.pypa.io/en/stable) via

```
pip install mario
```

## Examples

### Plot the Proton Spectra of a HESS source
The following example will show you how to plot the proton spectra of given counterpart to a HESS source.

```python
from mario.sources.sources import Source
hess = Source(name='HESS J1804', spectral_index=2.69, 
            flux_GeV=1.56e-9 / (u.s * u.cm**2),
            flux_E_GeV=0.01 * u.TeV,
            flux_TeV=5.3e-11/(u.s * u.cm**2),
            flux_E_TeV=0.2 * u.TeV,
            mass=7.9e5 * u.solMass)

from mario.sources.sources import SNR
G8 = SNR(name='SNR G8.7-0.1', 
            glon=8.744205 * u.deg, glat=-0.096288 * u.deg, radius=0.375*u.deg,
            age=15 * u.kyr, distance=4.5 * u.kpc)

import numpy as np
from mario.models.protonspectrum import ProtonSpectrum
ps = ProtonSpectrum(gamma_src=hess, snr=G8, nH=160*u.cm**(-3)) 
Ep = np.logspace(0, 6, 50) * u.GeV
pflux = ps.proton_flux(Ep, rcloud=12*u.pc)

import matplotlib.pyplot as plt
from mario.plotting.plotting import proton_spectrum, add_kcr_src, add_kcr

fig = plt.figure(figsize=(9, 12))
ax = fig.add_subplot(211)
G8.age = 15*u.kyr
ps.flux(Ep, rcloud=ps.dist_cloud)
fig, ax  = proton_spectrum(spectrum=ps, fig=fig, ax=ax, diff_label=True, color='#00CDCD')

G8.age = 28*u.kyr
ps.flux(Ep, rcloud=ps.dist_cloud)
fig, ax = proton_spectrum(spectrum=ps, fig=fig, ax=ax, color='black')
ax.tick_params(axis='x', which='both', labelbottom=False)
ax.set_xlabel('')
ax = add_kcr(label='CR Flux at Earth', ax=ax, ls='-.', lw=2, color='magenta')
ax = add_kcr_src(spectrum=ps, emin=60*u.GeV, emax=630*u.GeV, ax=ax, label='k (GeV)', 
                 color='green', ls='--', lw=2)
ax = add_kcr_src(spectrum=ps, emin=600*u.GeV, emax=8000*u.GeV, ax=ax, label='k (TeV)',
                 color='red', ls='--', lw=2)

ax = fig.add_subplot(212)
G8.age = 15*u.kyr
ps.diff_delta=0.5
ps.flux(Ep, rcloud=ps.dist_cloud)
fig, ax  = proton_spectrum(spectrum=ps, fig=fig, ax=ax, color='#00CDCD')

G8.age = 28*u.kyr
ps.diff_delta=0.5
ps.flux(Ep, rcloud=ps.dist_cloud)
fig, ax = proton_spectrum(spectrum=ps, fig=fig, ax=ax, diff_label=True, age_label=True, color='black')
ax = add_kcr(label='CR Flux at Earth', ax=ax, ls='-.', lw=2, color='magenta')
ax = add_kcr_src(spectrum=ps, emin=60*u.GeV, emax=630*u.GeV, ax=ax, label='k (GeV)', 
                 color='green', ls='--', lw=2)
ax = add_kcr_src(spectrum=ps, emin=600*u.GeV, emax=8000*u.GeV, ax=ax, label='k (TeV)',
                 color='red', ls='--', lw=2)
fig.subplots_adjust(hspace=0.08)

```


## Dependencies
`mario` is tested on Python 3.6 and depends on NumPy, SciPy, Matplotlib, GammaPy and spectral_cube
(see requirements.txt for version information).


## License
`mario` is released under the MIT License, see the LICENSE file for the full text.





