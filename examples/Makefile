SHELL=/bin/bash
build_dir = ./build
data_dir = /home/kirsty/Documents/PhD/Fits_Files/Models
hess = /home/kirsty/Documents/PhD/Fits_Files/HESS_2018

config_acc = ./accelerators.yaml
config_gam_srcs = ./gamma_sources.yaml
config_model = ./models.yaml
config_gamma_model = ./gamma_models.yaml
config_irf = ./irf.yaml
config_plot = ./plot.yaml


cube_pdf: $(config_acc) $(config_model)
	mario_create_pdf_cube 'G8' 'generic' ${data_dir)/total_col_dens_26_56.fits


cube_proton: $(config_model)
	mario_create_proton_cube '' 'generic'

cube_gamma: $(config_gamma_model)
	mario_create_gamma_cube 'G8' 'generic' ${data_dir)/total_col_dens_26_56.fits ${data_dir)/HadronicPDF.fits

plot_gamma: $(config_gamma_model)
	mario_plot_gamma_fluxes 'HESSJ1804' 'PJ' 'PJ_Plot' ../../data/HESS_data.txt ../../data/hgps_map_flux_0.1deg_v1_crop.fits '/home/kirsty/Documents/PhD/Fits_Files/Models/Protons/Gamma_cubes/Age16_PSR/HadronicFlux_AharonianImpulsiveTime_chi0.*'

spectra_plot: $(config_acc) $(config_model)
	mario_plot_spectral_flux 'HESSJ1804' HadronicFlux_AharonianImpulsiveTime_chi0.001_delta0.3_D03e+27_n160.0_n00.1_ESN1e+51_Emax100yr1.0_EmaxST150.0_tST15.0_deltap0.379.fits $(data_dir)/HESS_data.txt


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
