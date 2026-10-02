from pathlib import Path
import os

from OT_analysis import config
from OT_analysis import plotting
from OT_analysis import io_utils
from OT_analysis import workflow
from OT_analysis import analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'
DATA_DIR = PROJECT_ROOT / 'data'

def main():

	make_plots = False
	verbose = True
	save_dir = RESULTS_DIR / 'synthetic_data'
	# delete the plots/ directory and its contents if it exists
	io_utils.delete_plots_directory(save_dir)

	# ---------------------------------------------------------
	# Get parameters
	# ---------------------------------------------------------
	base_params = config.create_parameters()
	base_params['exp']['load_exp_data'] = False  # synthentic data is enfored here

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
	# Plot results
	# ---------------------------------------------------------
	if analysis_results['params_run']['plot']['comparing_parameters']:
		plotting.plot_parameter_comparison_singleTrace(
			analysis_results,
			relative_results,
			save_dir
		)
	

if __name__ == "__main__":
	main()
