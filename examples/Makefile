SHELL=/bin/bash
build_dir = ./build
config_acc = ./accelerators.yaml
config_gam_srcs = ./gamma_sources.yaml
config_model = ./models.yaml



all: $(config_acc) $(config_gam_srcs) $(config_model)
	mario_plot_proton_spectrum 'hess1804' 'G8'
	mario_plot_proton_spectrum 'hess1804' 'PSR1803'
	
clean:
	rm -rf $(build_dir)

$(build_dir):
	mkdir -p $(build_dir)
