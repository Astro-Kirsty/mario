SHELL=/bin/bash
build_dir = ./build
config_acc = ./accelerators.yaml
config_gam_srcs = ./gamma_sources.yaml
config_model = ./models.yaml
config_plot = ./plot.yaml


all: cube spectra_plot flux_plot


cube_G8: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_1' 'total_col_dens_26_56.fits'
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_2' 'total_col_dens_26_56.fits'
#	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_3' 'total_col_dens_26_56.fits'
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_4' 'total_col_dens_26_56.fits'
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_5' 'total_col_dens_26_56.fits'
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_6' 'total_col_dens_26_56.fits'
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_7' 'total_col_dens_26_56.fits'
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_8' 'total_col_dens_26_56.fits'
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_9' 'total_col_dens_26_56.fits'

cube_PSRJ1803: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p1' 'total_col_dens_8_26.fits'
	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p2' 'total_col_dens_8_26.fits'
#	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p3' 'total_col_dens_8_26.fits'
	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p4' 'total_col_dens_8_26.fits'
	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p5' 'total_col_dens_8_26.fits'
	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p6' 'total_col_dens_8_26.fits'
	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p7' 'total_col_dens_8_26.fits'
	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p8' 'total_col_dens_8_26.fits'
	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p9' 'total_col_dens_8_26.fits'


spectra_plot: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_plot_proton_spectrum 'HESSJ1804' 'G8'
	mario_plot_proton_spectrum 'HESSJ1804' 'PSRJ1803'
	mario_plot_electron_spectrum 'HESSJ1804' 'PSRJ1803'

int_flux_plot: $(config_acc) $(config_plot)
	mario_plot_integrated_flux Flux_G8_age15_delta0.3_chi0.1_E1e+50.fits 'G8' 'G8_Plot'
	mario_plot_integrated_flux Flux_PJ_age16_delta0.3_chi0.1_E1e+50.fits 'PSRJ1803' 'PSRJ1803_Plot'

spec_flux_plot: $(config_gam_srcs)
	mario_plot_spectral_flux Flux_G8_age15_delta0.3_chi0.1_E1e+50.fits 'HESSJ1804'

clean:
	rm -rf $(build_dir)

$(build_dir):
	mkdir -p $(build_dir)
