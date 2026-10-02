from pathlib import Path
import copy
import numpy as np
import time

from OT_analysis import config
from OT_analysis import plotting
from OT_analysis import io_utils
from OT_analysis import workflow
from OT_analysis import analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'
DATA_DIR = PROJECT_ROOT / 'data'


def simulation_ensemble_parameters(): # parameters specific to the ensemble analysis
	return {
		'sys':{
			'brownian_motion_type': 'OT1d_periodic_drive',  # {'OT1d', 'OT1d_periodic_drive'}
		},
		'sim': {
			'simulation_ensemble_random_seeds': True,  # if False, generates the same series of traces for reproducibility, following the index or range(); if True, generates a new random seed for each run
		},
		'ana': {
			'N_runs': 2,  # number of runs to perform for ensemble analysis
			'comparing_parameters_type': 'mean-std',  # {'mean-std', 'box', 'violin'}; 'mean-std' does simple mean & std, while 'box' does box plots, which are more informative, and 'violin' does violin plots, which are even more informative
			'use_tweeze_pkg': False,  # whether to use the tweeze package for analysis
		},	
	}

def configure_parameters(base_params, simulation_ensemble_params):
	base_params['exp']['load_exp_data'] = False  # synthetic data is enforced here
	base_params['sys']['brownian_motion_type'] = simulation_ensemble_params['sys']['brownian_motion_type']
	base_params['ana']['use_tweeze_pkg'] = simulation_ensemble_params['ana']['use_tweeze_pkg']

	if simulation_ensemble_params['sim']['simulation_ensemble_random_seeds']:
		base_params['sim']['generate_seed'] = True  # generate a new seed for each run
	else:
		base_params['sim']['generate_seed'] = False  # just use the same seed

	base_params['ana']['exclude_seg_smaller'] = 0  # include all segmentations
	base_params['ana']['drive_free_segments'] = True  # get peak from full trace
	base_params['ana']['fit_with_error'] = False
	base_params['ana']['exclude_aliasing_freqs'] = True  # exclude aliasing frequencies from the fit, by removing e.g. the last 20 points of the PSD and freq arrays

	if base_params['ana']['use_tweeze_pkg']:
		if not base_params['sys']['brownian_motion_type'] == 'OT1d':
			raise ValueError("Brownian motion must be without periodic drive when using the tweeze package.")
		
		base_params['use_tweeze']['load_tweeze_trace'] = False # technically one can do this, the trace however is just 1s, so the analysis is not very great

	return base_params


def organize_ensemble_results(base_params, all_results):

	# Stack arrays from all runs: shape becomes (n_runs, n_tested_window_sizes)\
	reshaped_results = {
		'base': {
			'rel_gamma':  np.stack([res['ana_results']['rel_gamma'] for res in all_results]),
			'rel_kappax': np.stack([res['ana_results']['rel_kappax'] for res in all_results]),
			'rel_d_bead': np.stack([res['ana_results']['rel_d_bead'] for res in all_results]),
			'rel_beta': np.stack([res['ana_results']['rel_beta'] for res in all_results])
		}
	}

	# summarize simulation_ensemble results: mean and std of the relative errors for each parameter
	simulation_ensemble_results = {
		'segmentation': {
			"n_seg": all_results[0]['n_seg_values'],  # same for all runs
			"t_win": all_results[0]['t_win_values'],  # same for all runs
			"n_in_win": all_results[0]['n_in_win_values'],  # same for all runs
		},
		'parameter_mean': {
			'rel_gamma': np.nanmean(reshaped_results['base']['rel_gamma'], axis=0),
			'rel_kappax': np.nanmean(reshaped_results['base']['rel_kappax'], axis=0),
			'rel_d_bead': np.nanmean(reshaped_results['base']['rel_d_bead'], axis=0),
			'rel_beta': np.nanmean(reshaped_results['base']['rel_beta'], axis=0),
		},
		'parameter_std': {
			'rel_gamma_std': np.nanstd(reshaped_results['base']['rel_gamma'], axis=0, ddof=1),
			'rel_kappax_std': np.nanstd(reshaped_results['base']['rel_kappax'], axis=0, ddof=1),
			'rel_d_bead_std': np.nanstd(reshaped_results['base']['rel_d_bead'], axis=0, ddof=1),
			'rel_beta_std': np.nanstd(reshaped_results['base']['rel_beta'], axis=0, ddof=1),
		},
	}

	# If the analysis includes tweezepy analysis, also summarize the results
	if base_params['ana']['use_tweeze_pkg'] and base_params['sys']['brownian_motion_type'] == 'OT1d':
		if base_params['use_tweeze']['do_psd']:

			reshaped_results['tweeze_psd'] = {
				'rel_gamma_psd': np.stack([res['ana_tweeze_results']['rel_gamma_psd'] for res in all_results]),
				'rel_kappax_psd': np.stack([res['ana_tweeze_results']['rel_kappax_psd'] for res in all_results]),
				'rel_d_bead_psd': np.stack([res['ana_tweeze_results']['rel_d_bead_psd'] for res in all_results]),
				'rel_beta_psd': np.stack([np.nan for res in all_results], axis=0)  # no beta analysis in tweezepy
			}

			simulation_ensemble_results['parameter_mean_tweeze'] = {
				'rel_gamma_psd': np.nanmean(reshaped_results['tweeze_psd']['rel_gamma_psd'], axis=0),
				'rel_kappax_psd': np.nanmean(reshaped_results['tweeze_psd']['rel_kappax_psd'], axis=0),
				'rel_d_bead_psd': np.nanmean(reshaped_results['tweeze_psd']['rel_d_bead_psd'], axis=0),
				'rel_beta_psd': np.nanmean(reshaped_results['tweeze_psd']['rel_beta_psd'], axis=0)  # no beta analysis in tweezepy
			}
			simulation_ensemble_results['parameter_std_tweeze'] = {
				'rel_gamma_psd_std': np.nanstd(reshaped_results['tweeze_psd']['rel_gamma_psd'], axis=0, ddof=1),
				'rel_kappax_psd_std': np.nanstd(reshaped_results['tweeze_psd']['rel_kappax_psd'], axis=0, ddof=1),
				'rel_d_bead_psd_std': np.nanstd(reshaped_results['tweeze_psd']['rel_d_bead_psd'], axis=0, ddof=1),
				'rel_beta_psd_std': np.nanstd(reshaped_results['tweeze_psd']['rel_beta_psd'], axis=0, ddof=1)  # no beta analysis in tweezepy
			}

		if base_params['use_tweeze']['do_AV']:

			reshaped_results['tweeze_AV'] = {
				'rel_gamma_AV': np.stack([res['ana_tweeze_results']['rel_gamma_AV'] for res in all_results]),
				'rel_kappax_AV': np.stack([res['ana_tweeze_results']['rel_kappax_AV'] for res in all_results]),
				'rel_d_bead_AV': np.stack([res['ana_tweeze_results']['rel_d_bead_AV'] for res in all_results]),
				'rel_beta_AV': np.stack([np.nan for res in all_results], axis=0)  # no beta analysis in tweezepy
			}

			parameter_mean_tweeze_AV = {
				'rel_gamma_AV': np.nanmean(reshaped_results['tweeze_AV']['rel_gamma_AV'], axis=0),
				'rel_kappax_AV': np.nanmean(reshaped_results['tweeze_AV']['rel_kappax_AV'], axis=0),
				'rel_d_bead_AV': np.nanmean(reshaped_results['tweeze_AV']['rel_d_bead_AV'], axis=0),
				'rel_beta_AV': np.nanmean(reshaped_results['tweeze_AV']['rel_beta_AV'], axis=0)  # no beta analysis in tweezepy
			}
			parameter_std_tweeze_AV = {
				'rel_gamma_AV_std': np.nanstd(reshaped_results['tweeze_AV']['rel_gamma_AV'], axis=0, ddof=1),
				'rel_kappax_AV_std': np.nanstd(reshaped_results['tweeze_AV']['rel_kappax_AV'], axis=0, ddof=1),
				'rel_d_bead_AV_std': np.nanstd(reshaped_results['tweeze_AV']['rel_d_bead_AV'], axis=0, ddof=1),
				'rel_beta_AV_std': np.nanstd(reshaped_results['tweeze_AV']['rel_beta_AV'], axis=0, ddof=1)  # no beta analysis in tweezepy
			}

			if 'parameter_mean_tweeze' in simulation_ensemble_results:
				simulation_ensemble_results['parameter_mean_tweeze'].update(parameter_mean_tweeze_AV)
			else:
				simulation_ensemble_results['parameter_mean_tweeze'] = parameter_mean_tweeze_AV

			if 'parameter_std_tweeze' in simulation_ensemble_results:
				simulation_ensemble_results['parameter_std_tweeze'].update(parameter_std_tweeze_AV)
			else:
				simulation_ensemble_results['parameter_std_tweeze'] = parameter_std_tweeze_AV


	return reshaped_results, simulation_ensemble_results


def plot_parameter_comparisons(
	reshaped_results,
	simulation_ensemble_results,
	base_params,
	simulation_ensemble_params,
	save_dir
):
		
	# plot the results of the simulation_ensemble analysis
	simulation_ensemble_ylabels = [
		r'$(\gamma_{fit}-\gamma_{soll})\;/\;\gamma_{soll}$',
		r'$(\kappa_{x,fit}-\kappa_{x,soll})\;/\;\kappa_{x,soll}$',
		r'$(d_{bead,fit}-d_{bead,soll})\;/\;d_{bead,soll}$',
		r'$(\beta_{fit}-\beta_{soll})\;/\;\beta_{soll}$',
	]

	for r, res_key in enumerate(simulation_ensemble_results['parameter_mean'].keys()):
		data = {'base': reshaped_results['base'][res_key],}
		means = {'base': simulation_ensemble_results['parameter_mean'][res_key],}
		stds = {'base': simulation_ensemble_results['parameter_std'][res_key + '_std'],}
		methods = {'base': True,}

		# include analysis using tweezepy
		if base_params['ana']['use_tweeze_pkg'] and base_params['sys']['brownian_motion_type'] == 'OT1d':
			if base_params['use_tweeze']['do_psd']:
				data['tweeze_psd'] = reshaped_results['tweeze_psd'][res_key + '_psd']
				means['tweeze_psd'] = simulation_ensemble_results['parameter_mean_tweeze'][res_key + '_psd']
				stds['tweeze_psd'] = simulation_ensemble_results['parameter_std_tweeze'][res_key + '_psd_std']
				methods['tweeze_psd'] = True
				
			if base_params['use_tweeze']['do_AV']:
				data['tweeze_AV'] = reshaped_results['tweeze_AV'][res_key + '_AV']
				means['tweeze_AV'] = simulation_ensemble_results['parameter_mean_tweeze'][res_key + '_AV']
				stds['tweeze_AV'] = simulation_ensemble_results['parameter_std_tweeze'][res_key + '_AV_std']
				methods['tweeze_AV'] = True

		plotting.plot_ensemble_results(
			simulation_ensemble_params['ana']['N_runs'],
			data,
			means,
			stds,
			methods,
			simulation_ensemble_params['ana']['comparing_parameters_type'],
			base_params['sys']['t_msr'],
			base_params['plot']['comparing_parameters_onTop'],
			simulation_ensemble_results['segmentation']['n_seg'],
			simulation_ensemble_results['segmentation']['t_win'],
			simulation_ensemble_ylabels[r],
			res_key,
			save_dir
		)


def main():
	start = time.perf_counter()

	make_plots=False
	verbose=False
	save_dir = RESULTS_DIR / 'synthetic_ensemble_data'
	# delete the plots/ directory and its contents if it exists
	io_utils.delete_plots_directory(save_dir)

	# ---------------------------------------------------------
	# Get & configure parameters
	# ---------------------------------------------------------
	base_params = config.create_parameters()
	simulation_ensemble_params = simulation_ensemble_parameters()
	base_params = configure_parameters(base_params, simulation_ensemble_params)

	# ---------------------------------------------------------
	# Run analysis
	# ---------------------------------------------------------
	all_results = []

	for i_run in range(simulation_ensemble_params['ana']['N_runs']):
		print(f"Running ensemble analysis for run {i_run+1}/{simulation_ensemble_params['ana']['N_runs']}...")
		params = copy.deepcopy(base_params)

		if not simulation_ensemble_params['sim']['simulation_ensemble_random_seeds']:
			params['sim']['seed'] = i_run  # as no seed is generated, use the run index as the seed for reproducibility

		analysis_results = workflow.run_analysis(
			params,
			make_plots=make_plots,
			verbose=verbose,
			save_dir=save_dir
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

		if 'ana_tweeze_results' in relative_results:
			result['ana_tweeze_results'] = (
				relative_results['ana_tweeze_results']
			)

		all_results.append(result)

	# ---------------------------------------------------------
	# Plot results
	# ---------------------------------------------------------
	# Organize results
	reshaped_results, simulation_ensemble_results = organize_ensemble_results(
		base_params, 
		all_results
	)

	plot_parameter_comparisons(
		reshaped_results,
		simulation_ensemble_results,
		base_params,
		simulation_ensemble_params,
		save_dir
	)

	# ---------------------------------------------------------
	# Save results
	# ---------------------------------------------------------
	io_utils.save_ensemble_results(
		all_results, 
		simulation_ensemble_params, 
		base_params, 
		save_dir)


	end = time.perf_counter()
	print("  > Elapsed runtime time = {:.2f}s".format((end - start)))
	
	
if __name__ == "__main__":
	main()


