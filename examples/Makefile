SHELL=/bin/bash
build_dir = ./build
config_acc = ./accelerators.yaml
config_gam_srcs = ./gamma_sources.yaml
config_model = ./models.yaml


all: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_plot_proton_spectrum 'HESSJ1804' 'G8'
	mario_plot_proton_spectrum 'HESSJ1804' 'PSRJ1803'

cube: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_1' 'total_col_dens_26_56.fits'
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_2' 'total_col_dens_26_56.fits'
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_3' 'total_col_dens_26_56.fits'

	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p1' 'total_col_dens_8_26.fits'
	mario_create_gamma_cube 'HESSJ1804' 'PSRJ1803' 'hadronic_p2' 'total_col_dens_8_26.fits'

plots: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_plot_proton_spectrum 'HESSJ1804' 'G8'
	mario_plot_proton_spectrum 'HESSJ1804' 'PSRJ1803'

clean:
	rm -rf $(build_dir)

$(build_dir):
	mkdir -p $(build_dir)
