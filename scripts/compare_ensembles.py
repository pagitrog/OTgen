from pathlib import Path
import numpy as np

from OT_analysis import plotting
from OT_analysis import io_utils


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'


def comparison_parameters():
	return {
		'load_dir': RESULTS_DIR / 'compare_ensembles_data/',
		'save_dir': RESULTS_DIR / 'compare_ensembles_data/',
		'load_dirs_labels': ('A', 'B'),

		# Options: 'mean-std', 'box', 'violin'
		'comparing_parameters_type': 'mean-std',

		'x_range_to_plot': (None, None),
		'y_range_to_plot': (None, None),
	}


def get_base_values(raw_results, parameter):
	"""
	Get one quantity for all runs.

	Plain fitted parameters are stored in parameter_means.
	Absolute and relative errors are stored in ana_results.
	"""

	plain_parameters = {
		'beta',
		'gamma',
		'kappa',
		'd_bead',
	}

	if parameter in plain_parameters:
		return np.stack([
			res['parameter_means'][parameter]
			for res in raw_results
		])

	return np.stack([
		res['ana_results'][parameter]
		for res in raw_results
	])


def get_tweeze_values(raw_results, parameter, method):
	"""
	Get Tweezepy quantities if they exist in the saved results.

	method:
		'psd'
		'AV'
	"""

	if parameter == 'beta':
		return None

	parameter_key = f'{parameter}_{method}'

	if not all(
		'ana_tweeze_results' in res
		and parameter_key in res['ana_tweeze_results']
		for res in raw_results
	):
		return None

	return np.stack([
		res['ana_tweeze_results'][parameter_key]
		for res in raw_results
	])


def get_methods(raw_results, parameter):
	"""
	Assemble base and, where available, Tweezepy results.

	Statistics are calculated here directly from all runs so that
	plain parameters, absolute errors and relative errors are treated
	in exactly the same way.
	"""

	base_values = get_base_values(
		raw_results,
		parameter,
	)

	methods = {
		'base': {
			'data': base_values,
			'mean': np.nanmean(base_values, axis=0),
			'median': np.nanmedian(base_values, axis=0),
			'std': np.nanstd(base_values, axis=0, ddof=1),
		},
	}

	tweeze_psd_values = get_tweeze_values(
		raw_results,
		parameter,
		'psd',
	)

	if tweeze_psd_values is not None:
		methods['tweeze_psd'] = {
			'data': tweeze_psd_values,
			'mean': np.nanmean(tweeze_psd_values, axis=0),
			'median': np.nanmedian(tweeze_psd_values, axis=0),
			'std': np.nanstd(tweeze_psd_values, axis=0, ddof=1),
		}

	tweeze_AV_values = get_tweeze_values(
		raw_results,
		parameter,
		'AV',
	)

	if tweeze_AV_values is not None:
		methods['tweeze_AV'] = {
			'data': tweeze_AV_values,
			'mean': np.nanmean(tweeze_AV_values, axis=0),
			'median': np.nanmedian(tweeze_AV_values, axis=0),
			'std': np.nanstd(tweeze_AV_values, axis=0, ddof=1),
		}

	return methods


def plot_parameter_comparisons(
	loaded_results,
	save_dir,
	comparing_parameters_type,
	x_range_to_plot,
	y_range_to_plot,
):

	ylabels = {
		# -----------------------------------------------------
		# Plain parameters
		# -----------------------------------------------------
		'gamma':
			r'$\gamma_{fit}$ [Ns/m]',

		'kappa':
			r'$\kappa_{x,fit}$ [N/m]',

		'd_bead':
			r'$d_{bead,fit}$ [m]',

		'beta':
			r'$\beta_{fit}$ [m/V]',

		# -----------------------------------------------------
		# Absolute errors
		# -----------------------------------------------------
		'abs_gamma':
			r'$(\gamma_{fit}-\gamma_{soll})$ [Ns/m]',

		'abs_kappax':
			r'$(\kappa_{x,fit}-\kappa_{x,soll})$ [N/m]',

		'abs_d_bead':
			r'$(d_{bead,fit}-d_{bead,soll})$ [m]',

		'abs_beta':
			r'$(\beta_{fit}-\beta_{soll})$ [m/V]',

		# -----------------------------------------------------
		# Relative errors
		# -----------------------------------------------------
		'rel_gamma':
			r'$(\gamma_{fit}-\gamma_{soll})\;/\;\gamma_{soll}$',

		'rel_kappax':
			r'$(\kappa_{x,fit}-\kappa_{x,soll})'
			r'\;/\;\kappa_{x,soll}$',

		'rel_d_bead':
			r'$(d_{bead,fit}-d_{bead,soll})'
			r'\;/\;d_{bead,soll}$',

		'rel_beta':
			r'$(\beta_{fit}-\beta_{soll})'
			r'\;/\;\beta_{soll}$',
	}

	for parameter, ylabel in ylabels.items():

		analyses = {}

		for analysis_label, results in loaded_results.items():

			raw_results = results['all_results']

			# Check whether this quantity exists in this ensemble.
			# This is useful, for example, for experimental data where
			# no reference value for beta or kappa may be available.
			try:
				methods = get_methods(
					raw_results,
					parameter,
				)

			except KeyError:
				print(
					f"> Skipping {parameter} for {analysis_label}: "
					"quantity not stored."
				)
				continue

			analyses[analysis_label] = {
				'N_runs': results['N_runs'],
				't_msr': results['t_msr'],
				't_win': results['segmentation']['t_win'],
				'methods': methods,
			}

		# Nothing available for this quantity.
		if not analyses:
			continue

		plotting.plot_compare_ensembles(
			analyses=analyses,
			ylabel=ylabel,
			title=parameter,
			save_dir=save_dir,
			comparing_parameters_type=comparing_parameters_type,
			x_range_to_plot=x_range_to_plot,
			y_range_to_plot=y_range_to_plot,
		)


def main():
	comparison_params = comparison_parameters()

	# ---------------------------------------------------------
	# Directories
	# ---------------------------------------------------------
	load_dir = comparison_params['load_dir']
	save_dir = comparison_params['save_dir']

	load_dirs = {
		label: load_dir / label
		for label in comparison_params['load_dirs_labels']
	}

	# ---------------------------------------------------------
	# Plot settings
	# ---------------------------------------------------------
	comparing_parameters_type = (
		comparison_params['comparing_parameters_type']
	)

	x_range_to_plot = comparison_params['x_range_to_plot']
	y_range_to_plot = comparison_params['y_range_to_plot']

	# ---------------------------------------------------------
	# Load saved ensembles
	# ---------------------------------------------------------
	loaded_results = {
		label: io_utils.load_ensemble_results(directory)
		for label, directory in load_dirs.items()
	}

	# ---------------------------------------------------------
	# Plot
	# ---------------------------------------------------------
	plot_parameter_comparisons(
		loaded_results,
		save_dir,
		comparing_parameters_type,
		x_range_to_plot,
		y_range_to_plot,
	)


if __name__ == "__main__":
	main()