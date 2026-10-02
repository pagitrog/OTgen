from pathlib import Path
import os
import pickle
import numpy as np

from OT_analysis import config
from OT_analysis import plotting
from OT_analysis import io_utils
from OT_analysis import workflow
from OT_analysis import analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'
DATA_DIR = PROJECT_ROOT / 'data'


def recover_parameters():
	return {
		'load_dir': RESULTS_DIR / "synthetic_ensemble_data/",
		'save_dir': RESULTS_DIR / "synthetic_ensemble_data/recovered_plots/",

		'x_range_to_plot': (None, None),
		'y_range_to_plot': (None, None),
	}

def plot_parameter_comparisons(
	results,
	all_results,
	ensemble_params,
	base_params,
	save_dir,
	x_range_to_plot,
	y_range_to_plot,
):
	ensemble_ylabels = {
		'rel_gamma': r'$(\gamma_{fit}-\gamma_{soll})\;/\;\gamma_{soll}$',
		'rel_kappax': r'$(\kappa_{x,fit}-\kappa_{x,soll})\;/\;\kappa_{x,soll}$',
		'rel_d_bead': r'$(d_{bead,fit}-d_{bead,soll})\;/\;d_{bead,soll}$',
		'rel_beta': r'$(\beta_{fit}-\beta_{soll})\;/\;\beta_{soll}$',
	}

	# Plot each parameter
	for res_key in results['base'].keys():

		# Base analysis
		base_values = np.stack([
			res['ana_results'][res_key]
			for res in all_results
		])

		data = {'base': base_values,}
		means = {'base': results['base'][res_key]['mean'],}
		stds = {'base':	results['base'][res_key]['std'],}
		methods = {'base': True,}

		# Tweezepy PSD
		if (res_key != 'rel_beta' and 'tweeze_psd' in results):

			tweeze_key = (res_key + '_psd')

			data['tweeze_psd'] = np.stack([
				res['ana_tweeze_results'][tweeze_key]
				for res in all_results
			])
			means['tweeze_psd'] = (results['tweeze_psd'][res_key]['mean'])
			stds['tweeze_psd'] = (results['tweeze_psd'][res_key]['std'])
			methods['tweeze_psd'] = True

		# Tweezepy AV
		if (res_key != 'rel_beta' and 'tweeze_AV' in results):

			tweeze_key = (res_key + '_AV')

			data['tweeze_AV'] = np.stack([
				res['ana_tweeze_results'][tweeze_key]
				for res in all_results
			])
			means['tweeze_AV'] = (results['tweeze_AV'][res_key]['mean'])
			stds['tweeze_AV'] = (results['tweeze_AV'][res_key]['std'])
			methods['tweeze_AV'] = True

		# Actual plot call
		plotting.plot_ensemble_results(
			results['N_runs'],
			data,
			means,
			stds,
			methods,
			ensemble_params['ana']['comparing_parameters_type'],
			results['t_msr'],
			base_params['plot']['comparing_parameters_onTop'],
			results['segmentation']['n_seg'],
			results['segmentation']['t_win'],
			ensemble_ylabels[res_key],
			res_key,
			save_dir,
			x_range_to_plot=x_range_to_plot,
			y_range_to_plot=y_range_to_plot,
		)


	
def main():
	simple_load_params = recover_parameters()

	# Load directory and save directory
	load_dir = simple_load_params['load_dir']
	save_dir = simple_load_params['save_dir']

	os.makedirs(save_dir, exist_ok=True)

	# Select x, y ranges to plot
	x_range_to_plot = simple_load_params['x_range_to_plot']
	y_range_to_plot = simple_load_params['y_range_to_plot']


	# ---------------------------------------------------------
	# Load saved data
	# ---------------------------------------------------------
	results = io_utils.load_ensemble_results(load_dir)

	all_results = results['all_results']
	base_params = results['base_params']

	with open(
		os.path.join(load_dir, "ensemble_params.pkl",), "rb",) as file:
		ensemble_params = pickle.load(file)

	# ---------------------------------------------------------
	# Plot results
	# ---------------------------------------------------------
	plot_parameter_comparisons(
		results,
		all_results,
		ensemble_params,
		base_params,
		save_dir,
		x_range_to_plot,
		y_range_to_plot,
	)

if __name__ == "__main__":
	main()