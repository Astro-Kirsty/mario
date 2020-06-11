SHELL=/bin/bash
build_dir = ./build
data_dir = /home/kirsty/Documents/PhD/GitHub/data
config_acc = ./accelerators.yaml
config_gam_srcs = ./gamma_sources.yaml
config_model = ./models.yaml
config_irf = ./irf.yaml
config_plot = ./plot.yaml


all: cube spectra_plot flux_plot

cube_G8: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_1' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_2' $(data_dir)/total_col_dens_26_56.fits
#	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_3' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_4' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_5' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_6' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_7' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_8' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_9' $(data_dir)/total_col_dens_26_56.fits

	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_11' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_12' $(data_dir)/total_col_dens_26_56.fits
#	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_13' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_14' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_15' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_16' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_17' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_18' $(data_dir)/total_col_dens_26_56.fits
	mario_create_gamma_cube 'HESSJ1804' 'G8' 'hadronic_19' $(data_dir)/total_col_dens_26_56.fits

cube_PJ: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_create_gamma_cube 'HESSJ1804' 'PJ' 'hadronic_p1' $(data_dir)/total_col_dens_8_26.fits
	mario_create_gamma_cube 'HESSJ1804' 'PJ' 'hadronic_p2' $(data_dir)/total_col_dens_8_26.fits
#	mario_create_gamma_cube 'HESSJ1804' 'PJ' 'hadronic_p3' $(data_dir)/total_col_dens_8_26.fits
	mario_create_gamma_cube 'HESSJ1804' 'PJ' 'hadronic_p4' $(data_dir)/total_col_dens_8_26.fits
	mario_create_gamma_cube 'HESSJ1804' 'PJ' 'hadronic_p5' $(data_dir)/total_col_dens_8_26.fits
	mario_create_gamma_cube 'HESSJ1804' 'PJ' 'hadronic_p6' $(data_dir)/total_col_dens_8_26.fits
	mario_create_gamma_cube 'HESSJ1804' 'PJ' 'hadronic_p7' $(data_dir)/total_col_dens_8_26.fits
	mario_create_gamma_cube 'HESSJ1804' 'PJ' 'hadronic_p8' $(data_dir)/total_col_dens_8_26.fits
	mario_create_gamma_cube 'HESSJ1804' 'PJ' 'hadronic_p9' $(data_dir)/total_col_dens_8_26.fits


spectra_plot: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_plot_proton_spectrum 'HESSJ1804' 'G8' 'generic'
	mario_plot_proton_spectrum 'HESSJ1804' 'PJ' 'generic'
	mario_plot_electron_spectrum 'HESSJ1804' 'PJ' 'generic'
	mario_plot_proton_spectrum 'HESSJ1804' 'G8' 'G8_timedependent' --timedependent True

int_flux_plot: $(config_acc) $(config_plot)
	mario_plot_integrated_flux 'G8' 'G8_Plot' $(data_dir)/Flux_G8_age28_delta0.3_chi0.1_E2.17e+51.fits
	mario_plot_integrated_flux 'PJ' 'PJ_Plot' $(data_dir)/Flux_PJ_age16_delta0.3_chi0.1_E1e+50.fits

spec_flux_plot: $(config_gam_srcs)
	mario_plot_spectral_flux 'HESSJ1804' $(data_dir)/Flux_G8_age28_delta0.3_chi0.1_E2.17e+51.fits

counts_plot: $(config_acc) $(config_plot) $(config_irf)
	mario_plot_counts 'G8' 'G8_Plot' 'HESS_irf' $(data_dir)/Flux_G8_age28_delta0.3_chi0.1_E2.17e+51.fits

clean:
	rm -rf $(build_dir)

$(build_dir):
	mkdir -p $(build_dir)
