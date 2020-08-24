SHELL=/bin/bash
build_dir = ./build
data_dir = /home/kirsty/Documents/PhD/Fits_Files/Models
hess = /home/kirsty/Documents/PhD/Fits_Files/HESS_2018

config_acc = ./accelerators.yaml
config_gam_srcs = ./gamma_sources.yaml
config_model = ./models.yaml
config_model_time = ./models_time.yaml
config_irf = ./irf.yaml
config_plot = ./plot.yaml


cube_proton: $(config_gam_srcs) $(config_gam_srcs) $(config_model)
	mario_create_proton_cube  'HESSJ1804' 'G8' 'hadronic_1' ${data_dir)/total_col_dens_26_56.fits

	mario_create_proton_cube  'HESSJ1804' 'PJ' 'hadronic_p1' ${data_dir)/total_col_dens_8_26.fits
	mario_create_proton_cube  'HESSJ1804' 'PJ' --model_config_path hadronic_time_models.yaml 'hadronic_p1' ${data_dir)/total_col_dens_8_26.fits


cube_G8: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age15_delta0.3_chi0.1_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age15_delta0.3_chi0.01_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age15_delta0.3_chi0.001_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age15_delta0.5_chi0.1_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age15_delta0.5_chi0.01_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age15_delta0.5_chi0.001_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age15_delta0.7_chi0.1_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age15_delta0.7_chi0.01_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age15_delta0.7_chi0.001_E1e+50.fits

	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age28_delta0.3_chi0.1_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age28_delta0.3_chi0.01_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age28_delta0.3_chi0.001_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age28_delta0.5_chi0.1_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age28_delta0.5_chi0.01_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age28_delta0.5_chi0.001_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age28_delta0.7_chi0.1_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age28_delta0.7_chi0.01_E1e+50.fits
	mario_create_gamma_cube 'G8' $(data_dir)/total_col_dens_26_56.fits $(data_dir)/ProtonFlux/ProtonFlux_G8_age28_delta0.7_chi0.001_E1e+50.fits


cube_PJ: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_create_gamma_cube 'PJ' $(data_dir)/total_col_dens_8_26.fits $(data_dir)/ProtonFlux/ProtonFlux_PJ_age16_delta0.3_chi0.1_E1e+50.fits
	mario_create_gamma_cube 'PJ' $(data_dir)/total_col_dens_8_26.fits $(data_dir)/ProtonFlux/ProtonFlux_PJ_age16_delta0.3_chi0.01_E1e+50.fits
	mario_create_gamma_cube 'PJ' $(data_dir)/total_col_dens_8_26.fits $(data_dir)/ProtonFlux/ProtonFlux_PJ_age16_delta0.3_chi0.001_E1e+50.fits
	mario_create_gamma_cube 'PJ' $(data_dir)/total_col_dens_8_26.fits $(data_dir)/ProtonFlux/ProtonFlux_PJ_age16_delta0.5_chi0.1_E1e+50.fits
	mario_create_gamma_cube 'PJ' $(data_dir)/total_col_dens_8_26.fits $(data_dir)/ProtonFlux/ProtonFlux_PJ_age16_delta0.5_chi0.01_E1e+50.fits
	mario_create_gamma_cube 'PJ' $(data_dir)/total_col_dens_8_26.fits $(data_dir)/ProtonFlux/ProtonFlux_PJ_age16_delta0.5_chi0.001_E1e+50.fits
	mario_create_gamma_cube 'PJ' $(data_dir)/total_col_dens_8_26.fits $(data_dir)/ProtonFlux/ProtonFlux_PJ_age16_delta0.7_chi0.1_E1e+50.fits
	mario_create_gamma_cube 'PJ' $(data_dir)/total_col_dens_8_26.fits $(data_dir)/ProtonFlux/ProtonFlux_PJ_age16_delta0.7_chi0.01_E1e+50.fits
	mario_create_gamma_cube 'PJ' $(data_dir)/total_col_dens_8_26.fits $(data_dir)/ProtonFlux/ProtonFlux_PJ_age16_delta0.7_chi0.001_E1e+50.fits


spectra_plot: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_plot_proton_spectrum 'HESSJ1804' 'G8' 'generic'
	mario_plot_proton_spectrum 'HESSJ1804' 'PJ' 'generic'
	mario_plot_electron_spectrum 'HESSJ1804' 'PJ' 'generic'
	mario_plot_proton_spectrum 'HESSJ1804' 'G8' 'time_hadronic'

int_flux_plot: $(config_acc) $(config_plot)
	mario_plot_integrated_flux 'G8' 'G8_Plot' $(data_dir)/Flux/Flux_G8_age28_delta0.3_chi0.1_E1e+50.fits
	mario_plot_integrated_flux 'PJ' 'PJ_Plot' $(data_dir)/Flux/Flux_PJ_age16_delta0.3_chi0.1_E1e+50.fits

residuals_plot: $(config_acc) $(config_gam_srcs) $(config_plot)
	mario_plot_residuals 'G8' 'HESSJ1804' 'G8_Plot' $(data_dir)/IntFlux/FluxInt_E1-100_G8_age15_delta0.3_chi0.1_E1e+50.fits $(hess)/hgps_map_flux_0.1deg_v1_crop.fits --hgps_catalog $(hess)/hgps_catalog_v1.fits

spec_flux_plot: $(config_gam_srcs)
	mario_plot_spectral_flux 'HESSJ1804' $(data_dir)/Flux/Flux_G8_age28_delta0.3_chi0.1_E1e+50.fits $(data_dir)/FGES_data.txt $(data_dir)/HESS_data.txt

counts_plot: $(config_acc) $(config_plot) $(config_irf)
	mario_plot_counts 'G8' 'G8_Plot' 'HESS_irf' $(data_dir)/Flux/Flux_G8_age28_delta0.3_chi0.1_E1e+50.fits

clean:
	rm -rf $(build_dir)

$(build_dir):
	mkdir -p $(build_dir)
