"""
Setup file for the mario package.
"""

from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='mario', 
    description='Modelling gAlactic gamma-Ray emissIOn',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Astro-Kirsty/mario',
    author='Kirsty Feijen',
    author_email='kirsty.feijen@adelaide.edu.au',
    license='MIT',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'spectral_cube>=0.4.5',
        'gammapy~=0.16',
        'astropy',
        'click',
        ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'mario_create_pdf_cube = mario.scripts.create_pdf_cube:main',
            'mario_create_gamma_cube = mario.scripts.create_gamma_cube:main',
            'mario_plot_gamma_fluxes = mario.scripts.plot_gamma_fluxes:main',
        ],
    }          
)
