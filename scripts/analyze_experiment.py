from pathlib import Path
import sys
import scipy.constants as const
import numpy as np

from OT_analysis import config
from OT_analysis import plotting
from OT_analysis import io_utils
from OT_analysis import workflow
from OT_analysis import analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'
DATA_DIR = PROJECT_ROOT / 'data'


def experimental_parameters():
	return {
		'exp': {
			'load_exp_data': True,  # whether to load experimental data or not, if FALSE, generates synthetic data
			'load_exp_data_dir': DATA_DIR / 'experimental/Sylvain/260325/',  # path to the experimental data file, if applicable
			'trace_i': 0,  # index of the trace to load from the directory
			'data_labels': ['x', 'y', 'z', 'x_stage', 'y_stage'],  
			'coordinate_of_drive': 'x',  # either 'x', 'y', or 'z', should be present as a stage coordinate, e.g. cannot be z if no z_stage
		},
		'sys': { # Corresponding to SiO2 beads
			'brownian_motion_type': 'OT1d_periodic_drive',  # {'OT1d', 'OT1d_periodic_drive'}

			't_msr': None,  # s; total simulation time; if None => extracted from load_exp_data
			'dt_sample': None,  # s; sampling time for simulation data; if None => extracted from load_exp_data

			'd_bead': .5e-6,  # m; bead diameter
			'm_bead': 4/3 * const.pi * (.5e-6/2)**3 * 1.05e3,  # kg; bead mass, assuming density of silica
			'T': 295.15,  # K; temperature
			'eta': .9544e-3,  # Ns/m2; viscosity, at T=22.Celsius
			'kappax': None,  # N/m; trap-stiffness

			'f_drive': None,  # 1/s; periodic drive frequency
			'A_drive': 50e-9,  # m; periodic drive amplitude
			'beta': None,  # m/V; detector sensitivity
		},
		# 'ana':{
		# 	# other packages
		# 	'use_tweeze_pkg': True, # note that if True, generating a trace from tweeze has higher priority
		# }
	}



def main():

	save_dir = RESULTS_DIR / 'experimental_data'
	make_plots = True
	verbose = True
	io_utils.delete_plots_directory(save_dir)

	# Set base parameters
	base_params = config.create_parameters()
	base_params.update(experimental_parameters())

	# ---------------------------------------------------------
	# Run analysis
	# ---------------------------------------------------------
	analysis_results = workflow.run_analysis(
		base_params,
		make_plots=make_plots,
		verbose=verbose,
		save_dir=save_dir
	)

	# get the relative errors of the parameters
	relative_results = analysis.get_relative_parameters(
		analysis_results
	)


	# ---------------------------------------------------------
	# Plot compared parameters
	# ---------------------------------------------------------
	if analysis_results['params_run']['plot']['comparing_parameters']:
		plotting.plot_parameter_comparison_singleTrace(
			analysis_results,
			relative_results,
			save_dir
		)


if __name__ == "__main__":
	main()