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
	mario_create_pdf_cube 'G8_15' 's1_G8_15' ${data_dir)/total_col_dens_26_56.fits
	mario_create_pdf_cube 'G8_28' 's1_G8_28' ${data_dir)/total_col_dens_26_56.fits
	mario_create_pdf_cube 'PJ_16' 's1_PJ_16' ${data_dir)/total_col_dens_8_26.fits

cube_gamma: $(config_gamma_model)
	mario_create_gamma_cube 'HESSJ1804' 'generic1' 'G8_15' 'G8_Plot' ${data_dir)/total_col_dens_26_56.fits ${data_dir}/hgps_map_flux_0.1deg_v1_crop.fits ${data_dir)/HadronicPDF.fits


plot_gamma: $(config_gamma_model)
	mario_plot_gamma_fluxes 'HESSJ1804' 'PJ_16' 'PJ_Plot' ../../data/HESS_data.txt ../../data/hgps_map_flux_0.1deg_v1_crop.fits '/home/kirsty/Documents/PhD/Fits_Files/Models/Protons/Gamma_cubes/PJ_16kyr/HadronicFlux_AharonianImpulsiveTime_chi0.*'


clean:
	rm -rf $(build_dir)

$(build_dir):
	mkdir -p $(build_dir)
