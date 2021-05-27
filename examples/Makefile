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
	mario_create_pdf_cube.py 'G8_15' 'HESSJ1804' --model_config_path hadronic_G8.yaml 's1_G8_15' ${data_dir/data/total_col_dens_26_56.fits
	mario_create_pdf_cube 'G8_28' 'HESSJ1804' --model_config_path hadronic_G8.yaml 's1_G8_28' ${data_dir/data/total_col_dens_26_56.fits
	mario_create_pdf_cube 'PJ_16' 'HESSJ1804' --model_config_path hadronic_G8.yaml 's1_PJ_16' ${data_dir/data/total_col_dens_8_26.fits

cube_gamma: $(config_gamma_model)
	mario_create_gamma_cube 'HESSJ1804' 'generic1' 'G8_15' 'G8_Plot' HadronicPDF.fits


plot_gamma: $(config_gamma_model)
	mario_plot_gamma_fluxes.py 'HESSJ1804' 'G8_15' 'G8_Plot' HadronicFlux_AharonianImpulsiveTimeExtent_chi0.1_delta0.3_D03e+27_n208_n00.03_ESN1e+51_EmaxST01.0_tST0234.85_deltap0.2.fits


clean:
	rm -rf $(build_dir)

$(build_dir):
	mkdir -p $(build_dir)
