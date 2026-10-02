from pathlib import Path
import copy
import time
import numpy as np
import scipy.constants as const

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
			'load_exp_data': True,
			'load_exp_data_dir': DATA_DIR / 'experimental/Sylvain/251208_1um/',
			'trace_i': 0,
			'data_labels': ['x', 'y', 'z', 'x_stage', 'y_stage'],
			'coordinate_of_drive': 'x',
		},
		'sys': {
			'brownian_motion_type': 'OT1d_periodic_drive',

			't_msr': None,
			'dt_sample': None,

			'd_bead': 977e-9,
			'm_bead': 4/3 * const.pi * (977e-9/2)**3 * 1.05e3,
			'T': 295.15,
			'eta': 1.e-3,
			'kappax': None,

			'f_drive': None,
			'A_drive': None,
			'beta': None,
		},
	}


def experimental_ensemble_parameters():
	return {
		'exp': {
			# None -> use all TDMS files in load_exp_data_dir
			'trace_indices': None,

			# Allowed relative deviation of the fitted drive frequency
			# from trace 1.
			'f_drive_rtol': 1e-3,
		},
		'ana': {
			'comparing_parameters_type': 'mean-std',
			'use_tweeze_pkg': False,
		},
	}


def configure_parameters(base_params, ensemble_params):
	base_params['exp']['load_exp_data'] = True
	base_params['ana']['use_tweeze_pkg'] = ensemble_params['ana']['use_tweeze_pkg']

	base_params['ana']['exclude_seg_smaller'] = 0
	base_params['ana']['drive_free_segments'] = True
	base_params['ana']['fit_with_error'] = False
	base_params['ana']['exclude_aliasing_freqs'] = True

	return base_params


def get_trace_indices(base_params, ensemble_params):
	trace_indices = ensemble_params['exp']['trace_indices']

	if trace_indices is not None:
		return list(trace_indices)

	data_dir = Path(base_params['exp']['load_exp_data_dir'])
	tdms_files = sorted(data_dir.glob('*.tdms'))

	if not tdms_files:
		raise FileNotFoundError(
			f"No TDMS files found in '{data_dir}'."
		)

	return list(range(len(tdms_files)))


def get_ensemble_metadata(base_params, ensemble_params, trace_indices):
	"""
	Inspect all experimental traces before the PSD analysis.

	The ensemble must have the same sampling interval and drive
	frequency. The common measurement time is the longest integer
	number of drive periods that fits into every trace.
	"""

	metadata = []

	for trace_i in trace_indices:
		params = copy.deepcopy(base_params)
		params['exp']['trace_i'] = trace_i

		exp_data, params = io_utils.get_exp_data_and_metadata(params)

		coordinate = params['exp']['coordinate_of_drive']
		n_samples = len(exp_data[coordinate])

		metadata.append({
			'trace_i': trace_i,
			'dt_sample': params['sys']['dt_sample'],
			'f_drive': params['sys']['f_drive'],
			'f_drive_fit': params['sys']['f_drive_fit'],
			'n_samples': n_samples,
		})

	# ---------------------------------------------------------
	# Validate common sampling interval
	# ---------------------------------------------------------
	dt_sample = metadata[0]['dt_sample']

	for item in metadata[1:]:
		if not np.isclose(item['dt_sample'], dt_sample):
			raise ValueError(
				"Experimental ensemble has inconsistent sampling intervals: "
				f"trace {trace_indices[0]} has dt_sample={dt_sample:.8g} s, "
				f"trace {item['trace_i']} has "
				f"dt_sample={item['dt_sample']:.8g} s."
			)

	# ---------------------------------------------------------
	# Validate common drive frequency
	# ---------------------------------------------------------
	f_drive = metadata[0]['f_drive']
	f_drive_fit = metadata[0]['f_drive_fit']
	f_drive_rtol = ensemble_params['exp']['f_drive_rtol']

	for item in metadata[1:]:
		if not np.isclose(
			item['f_drive_fit'],
			f_drive_fit,
			rtol=f_drive_rtol,
			atol=0.0,
		):
			raise ValueError(
				"Experimental ensemble has inconsistent fitted drive "
				"frequencies: "
				f"trace {trace_indices[0]} has "
				f"f_drive_fit={f_drive_fit:.8g} Hz, "
				f"trace {item['trace_i']} has "
				f"f_drive_fit={item['f_drive_fit']:.8g} Hz."
			)

		if not np.isclose(item['f_drive'], f_drive):
			raise ValueError(
				"Experimental traces do not resolve to the same "
				"FFT-compatible drive frequency."
			)

	# ---------------------------------------------------------
	# Determine common measurement time
	# ---------------------------------------------------------
	n_samples_per_period = int(
		np.round(1.0 / (f_drive * dt_sample))
	)

	n_periods_each = np.array([
		item['n_samples'] // n_samples_per_period
		for item in metadata
	])

	n_periods = int(np.min(n_periods_each))

	if n_periods < 1:
		raise ValueError(
			"The shortest experimental trace contains less than "
			"one complete drive period."
		)

	n_samples = n_periods * n_samples_per_period
	t_msr = n_samples * dt_sample

	# ---------------------------------------------------------
	# Construct common segmentation grid
	# ---------------------------------------------------------
	t_win_values, n_seg_values, n_in_win_values = (
		analysis.get_pairs_nseg_twin(
			base_params['ana']['seg_order_min'],
			base_params['ana']['seg_order_max'],
			base_params['ana']['n_seg_tests'],
			t_msr,
			dt_sample,
			1.0 / f_drive,
			4,
			base_params['ana']['exclude_seg_smaller'],
		)
	)

	if len(t_win_values) == 0:
		raise ValueError(
			"The common usable measurement time is too short to "
			"construct any valid segmentation pairs. "
			f"Common t_msr={t_msr:.6g} s "
			f"({n_periods} drive periods)."
		)

	return {
		'dt_sample': dt_sample,
		'f_drive': f_drive,
		't_msr': t_msr,
		'n_samples': n_samples,
		'n_samples_per_period': n_samples_per_period,
		'n_periods': n_periods,
		't_win_values': t_win_values,
		'n_seg_values': n_seg_values,
		'n_in_win_values': n_in_win_values,
	}


def organize_ensemble_results(all_results):

	result_keys = (
		'beta',
		'abs_beta',
		'rel_beta',

		'gamma',
		'abs_gamma',
		'rel_gamma',

		'kappa',
		'abs_kappax',
		'rel_kappax',

		'd_bead',
		'abs_d_bead',
		'rel_d_bead',
	)

	reshaped_results = {}

	for res_key in result_keys:

		if res_key in ('beta', 'gamma', 'kappa', 'd_bead'):
			values = np.stack([
				result['parameter_means'][res_key]
				for result in all_results
			])

		else:
			values = np.stack([
				result['ana_results'][res_key]
				for result in all_results
			])

		reshaped_results[res_key] = values

	ensemble_results = {
		'segmentation': {
			'n_seg': all_results[0]['n_seg_values'],
			't_win': all_results[0]['t_win_values'],
			'n_in_win': all_results[0]['n_in_win_values'],
		},
		'parameter_mean': {},
		'parameter_std': {},
	}

	for res_key, values in reshaped_results.items():

		ensemble_results['parameter_mean'][res_key] = np.nanmean(
			values,
			axis=0,
		)

		ensemble_results['parameter_std'][res_key + '_std'] = np.nanstd(
			values,
			axis=0,
			ddof=1,
		)

	return reshaped_results, ensemble_results


def plot_parameter_comparisons(
	reshaped_results,
	ensemble_results,
	base_params,
	ensemble_params,
	save_dir,
):

	ylabels = {
		'beta': r'$\beta_{fit}$ [m/V]',
		'abs_beta': r'$(\beta_{fit}-\beta_{soll})$ [m/V]',
		'rel_beta': r'$(\beta_{fit}-\beta_{soll})\;/\;\beta_{soll}$',

		'gamma': r'$\gamma_{fit}$ [Ns/m]',
		'abs_gamma': r'$(\gamma_{fit}-\gamma_{soll})$ [Ns/m]',
		'rel_gamma': r'$(\gamma_{fit}-\gamma_{soll})\;/\;\gamma_{soll}$',

		'kappa': r'$\kappa_{x,fit}$ [N/m]',
		'abs_kappax': r'$(\kappa_{x,fit}-\kappa_{x,soll})$ [N/m]',
		'rel_kappax': (
			r'$(\kappa_{x,fit}-\kappa_{x,soll})'
			r'\;/\;\kappa_{x,soll}$'
		),

		'd_bead': r'$d_{bead,fit}$ [m]',
		'abs_d_bead': (
			r'$(d_{bead,fit}-d_{bead,soll})$ [m]'
		),
		'rel_d_bead': (
			r'$(d_{bead,fit}-d_{bead,soll})'
			r'\;/\;d_{bead,soll}$'
		),
	}

	for res_key, ylabel in ylabels.items():

		data = {
			'base': reshaped_results[res_key],
		}

		means = {
			'base': ensemble_results[
				'parameter_mean'
			][res_key],
		}

		stds = {
			'base': ensemble_results[
				'parameter_std'
			][res_key + '_std'],
		}

		methods = {
			'base': True,
		}

		plotting.plot_ensemble_results(
			ensemble_params['ana']['N_runs'],
			data,
			means,
			stds,
			methods,
			ensemble_params['ana']['comparing_parameters_type'],
			base_params['sys']['t_msr'],
			base_params['plot']['comparing_parameters_onTop'],
			ensemble_results['segmentation']['n_seg'],
			ensemble_results['segmentation']['t_win'],
			ylabel,
			res_key,
			save_dir,
		)


def main():
	start = time.perf_counter()

	make_plots = False
	verbose = False
	save_dir = RESULTS_DIR / 'experimental_ensemble_data'

	io_utils.delete_plots_directory(save_dir)

	# ---------------------------------------------------------
	# Get & configure parameters
	# ---------------------------------------------------------
	base_params = config.create_parameters()
	base_params.update(experimental_parameters())

	ensemble_params = experimental_ensemble_parameters()
	base_params = configure_parameters(
		base_params,
		ensemble_params,
	)

	trace_indices = get_trace_indices(
		base_params,
		ensemble_params,
	)

	ensemble_params['ana']['N_runs'] = len(trace_indices)
	ensemble_params['exp']['trace_indices'] = trace_indices

	# ---------------------------------------------------------
	# Resolve common ensemble metadata
	# ---------------------------------------------------------
	ensemble_metadata = get_ensemble_metadata(
		base_params,
		ensemble_params,
		trace_indices,
	)

	print(
		f"> Experimental ensemble: {len(trace_indices)} traces, "
		f"f_drive={ensemble_metadata['f_drive']:.6g} Hz, "
		f"t_msr={ensemble_metadata['t_msr']:.6g} s, "
		f"{ensemble_metadata['n_periods']} drive periods."
	)

	resolved_base_params = copy.deepcopy(base_params)

	resolved_base_params['sys']['dt_sample'] = (
		ensemble_metadata['dt_sample']
	)
	resolved_base_params['sys']['f_drive'] = (
		ensemble_metadata['f_drive']
	)
	resolved_base_params['sys']['t_msr'] = (
		ensemble_metadata['t_msr']
	)

	# ---------------------------------------------------------
	# Run analysis
	# ---------------------------------------------------------
	all_results = []

	for i_run, trace_i in enumerate(trace_indices):

		print(
			f"Running experimental ensemble analysis for trace "
			f"{trace_i} ({i_run + 1}/{len(trace_indices)})..."
		)

		params = copy.deepcopy(base_params)
		params['exp']['trace_i'] = trace_i

		analysis_results = workflow.run_analysis(
			params,
			make_plots=make_plots,
			verbose=verbose,
			save_dir=save_dir,
			analysis_metadata=ensemble_metadata,
		)

		relative_results = analysis.get_relative_parameters(
			analysis_results
		)

		result = {
			'parameter_means': analysis_results['parameter_means'],
			'ana_results': relative_results['ana_results'],
			'n_seg_values': analysis_results['n_seg_values'],
			't_win_values': analysis_results['t_win_values'],
			'n_in_win_values': analysis_results['n_in_win_values'],
		}

		all_results.append(result)

	# ---------------------------------------------------------
	# Plot ensemble results
	# ---------------------------------------------------------
	reshaped_results, ensemble_results = (
		organize_ensemble_results(all_results)
	)

	plot_parameter_comparisons(
		reshaped_results,
		ensemble_results,
		resolved_base_params,
		ensemble_params,
		save_dir,
	)

	# ---------------------------------------------------------
	# Save in same format as simulation ensemble
	# ---------------------------------------------------------
	io_utils.save_ensemble_results(
		all_results,
		ensemble_params,
		resolved_base_params,
		save_dir,
	)

	end = time.perf_counter()

	print(
		"> Elapsed runtime time = {:.2f}s".format(
			end - start
		)
	)


if __name__ == "__main__":
	main()