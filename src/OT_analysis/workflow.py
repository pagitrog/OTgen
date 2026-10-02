import time
import sys
import copy
import numpy as np

from . import physics
from . import simulation
from . import analysis
from . import plotting
from . import io_utils


def prepare_run(base_params, printlog):

	# Deepcopy parameters
	params_run = copy.deepcopy(base_params)  # make a runtime copy of the parameters to avoid modifying the original

	# If experimental data, parameters may not be provided
	if params_run['sys']['beta'] is None:
		params_run['sys']['beta'] = np.nan
	if params_run['sys']['kappax'] is None:
		params_run['sys']['kappax'] = np.nan
	if params_run['sys']['d_bead'] is None:
		params_run['sys']['d_bead'] = np.nan

	# Calculate derived parameters
	parDerived = physics.calculate_derived_parameters(params_run)

	# Select segmentation-pair version
	if not params_run['ana']['create_pairs_version_override']:
		if params_run['sys']['brownian_motion_type'] == 'OT1d':
			params_run['ana']['create_pairs_nseg_twin_version'] = 3
		elif params_run['sys']['brownian_motion_type'] == 'OT1d_periodic_drive':
			params_run['ana']['create_pairs_nseg_twin_version'] = 4

	# Validate simulation parameters
	if not params_run['exp']['load_exp_data']:
		if params_run['sys']['t_msr'] <= 0 or params_run['sim']['dt_sim'] <= 0 or params_run['sys']['dt_sample'] <= 0:
			raise ValueError("t_msr, dt_sim, dt_sample must be positive.")
		if params_run['sim']['dt_sim'] > params_run['sys']['dt_sample']:
			raise ValueError("dt_sim must be smaller than dt_sample")
	
	# Configure Tweezepy
	if params_run['ana']['use_tweeze_pkg']:
		if params_run['sys']['brownian_motion_type'] == 'OT1d':
			params_run['sys']['beta'] = np.nan

			# if do_psd is set to True, note that the analysis is performed on top of the regular analysis
			if params_run['use_tweeze']['do_psd']:
				printlog("> Note: 'do_psd' is set to True. The PSD from tweezepy analysis will be performed in addition to the regular analysis.")	
			if params_run['use_tweeze']['do_AV']:
				printlog("> Note: 'do_AV' is set to True. The Allan Variance from tweezepy analysis will be performed in addition to the regular analysis.")
		else:
			printlog("> Warning: 'tweeze' package only supports 'OT1d' brownian_motion_type. Disabling tweeze analysis.")
			params_run['ana']['use_tweeze_pkg'] = False

		if not params_run['exp']['load_exp_data']:
			# if both load and generate tweeze trace are set to True, prioritize generating a new trace
			if params_run['use_tweeze']['generate_tweeze_trace'] and params_run['use_tweeze']['load_tweeze_trace']:
				printlog("> Warning: Both 'generate_tweeze_trace' and 'load_tweeze_trace' are set to True. Prioritizing generating a new trace.")
				params_run['use_tweeze']['load_tweeze_trace'] = False


	# # characteristic timescales
	# ## validate simulation time-step against the dynamic relaxation time
	# if params_run['sim']['dt_sim'] >= 0.01 * parDerived['sys']['phix']:
	# 	raise ValueError("Use dt << gamma/k, e.g. dt < 0.01 * gamma/k.")

	# if make_plots:  # the analysis and plot are redundant if we don't plot it

	# 	# do some time-scale analysis to investigate if dt_sim is small enough to resolve the dynamics of the system
	# 	if params_run['sim']['dt_sim'] <= 100 * parDerived['sys']['tau']:

	# 		# give a warning if dt_sim is about the same order of magnitude as the momentum relaxation time tau
	# 		printlog("> Warning: dt_sim is within two orders of magnitude of the momentum relaxation time tau. Do analysis.")

	# 		start = time.perf_counter()

	# 		# run trace analysis
	# 		trace_analysis_results = analysis.run_trace_analysis(
	# 			parDerived,
	# 			params_run
	# 		)

	# 		# plot trace analysis
	# 		plotting.plot_trace_analysis(
	# 			trace_analysis_results,
	# 			parDerived['sys']['tau'],
	# 			params_run['sim']['dt_sim'],
	# 			params_run['phys']['kB']*params_run['sys']['T']/params_run['sys']['m_bead'],	
	# 			save_dir=save_dir / 'plots',
	# 		)
	# 		end = time.perf_counter()
	# 		printlog("  > Elapsed momentum time-scale analysis time = {:.2f}s".format((end - start)))

	# 		if params_run['ana']['do_only_trace_analysis']:
	# 			sys.exit()

	return params_run, parDerived


def get_trajectory(parDerived, params_run, printlog):
	# get the seed for the simulation, either from the config file or generate a new one
	if params_run['sim']['generate_seed']:
		seed_sim = np.random.randint(0, 2**32 - 1)
	else:
		seed_sim = params_run['sim']['seed']

	np.random.seed(seed_sim)
	printlog(f"Using seed {seed_sim} for the simulation.")

	noise = np.random.normal(size=int(np.ceil((params_run['sys']['t_msr']) / params_run['sim']['dt_sim'])))
	
	t_sample, x_sample = simulation.get_traj_OT1d_overdamped(
		params_run['sim']["dt_sim"], # use dt_sim/1e2 to
		params_run['sys']['dt_sample'], # use t_msr = dt_sim
		params_run['sys']['t_msr'], # use t_msr = dt_sim
		parDerived['sim']['x_init'],
		noise,
		parDerived['sys']["phix"],
		parDerived['sys']["D"],
		params_run['sys']['f_drive'],
		params_run['sys']['A_drive'],
		params_run['sys']['brownian_motion_type'],
		params_run['sim']['sample_trajectory'],
	)
	return t_sample, x_sample, seed_sim


def get_sampled_trace(
	params_run,
	parDerived,
	printlog,
	save_dir,
	analysis_metadata=None,
):
	# get sampled data
	if params_run['exp']['load_exp_data']:

		# Get experimental data and update metadata (dt_sample, t_msr, f_drive, A_drive)
		exp_data, params_run = io_utils.get_exp_data_and_metadata(
			params_run=params_run
		)

		# for an ensemble of data
		if analysis_metadata is not None:
			if not np.isclose(
				params_run['sys']['dt_sample'],
				analysis_metadata['dt_sample'],
			):
				raise ValueError(
					"Experimental trace has a different dt_sample "
					"from the ensemble."
				)
			if not np.isclose(
				params_run['sys']['f_drive'],
				analysis_metadata['f_drive'],
			):
				raise ValueError(
					"Experimental trace has a different f_drive "
					"from the ensemble."
				)
			params_run['sys']['f_drive'] = analysis_metadata['f_drive']


		# Plot for verification : periodic vs non-periodic
		plotting.plot_exp_data(
			exp_data, 
			params_run, 
			save_dir
		)
		# truncate the data, so that t_msr is an integer multiple of 1/f_drive 
		t_msr_target = (
			analysis_metadata['t_msr']
			if analysis_metadata is not None
			else None
		)

		exp_data, params_run = analysis.truncate_exp_data_to_drive_periods(
			exp_data,
			params_run,
			t_msr_target=t_msr_target,
		)

		# Note: The data is experimental -> No unit conversion is applied
		if params_run['sys']['brownian_motion_type'] == 'OT1d_periodic_drive':
			coordinate = params_run['exp']['coordinate_of_drive']
			x_sample = exp_data[coordinate]
		else:
			x_sample = exp_data['x']
		t_sample = exp_data['t']

		seed_sim = None

	else:
		start = time.perf_counter()

		if params_run['ana']['use_tweeze_pkg'] and params_run['use_tweeze']['generate_tweeze_trace']:
			from tweezepy import simulate_trace

			if params_run['sim']['generate_seed']:
				seed_sim = np.random.randint(0, 2**32 - 1)
			else:
				seed_sim = params_run['sim']['seed']

			len_x_sample = int(params_run['sys']['t_msr']/params_run['sys']['dt_sample'])

			x_sample = simulate_trace(
				gamma=parDerived['sys']['gamma']*1e3,  # they state it as pNs/nm^2 ...  -> take it as fNs/nm
				kappa=params_run['sys']['kappax']*1e3,  # they state it as pN/nm ... -> take it as fN/nm
				fsim=1/params_run['sys']['dt_sample'], 
				sim_points=len_x_sample+1, 
				seed=seed_sim
			)
			x_sample = np.array(x_sample) * 1e-9 # convert nm to m for our code
			t_sample = np.arange(len_x_sample) * params_run['sys']['dt_sample']

		elif params_run['ana']['use_tweeze_pkg'] and params_run['use_tweeze']['load_tweeze_trace']:
			from tweezepy import load_trajectory

			x_sample = np.array(load_trajectory()) * 1e-9  # convert nm to m for our code
			t_sample = np.arange(len(x_sample)) * params_run['sys']['dt_sample']

			params_run['sys']['t_msr'] = t_sample[-1] + params_run['sys']['dt_sample']  # update t_msr based on the loaded trace
			seed_sim = None  # seed is not applicable when loading a trace

		else:
			t_sample, x_sample, seed_sim = get_trajectory(parDerived, params_run, printlog)

		end = time.perf_counter()
		printlog("  > Elapsed simulation time = {:.2f}s".format((end - start)))

		# Note: The data is synthetic (by default in m) -> If a beta is given, convert from m to volts
		if params_run['sys']['beta'] is not np.nan and params_run['sys']['brownian_motion_type']=='OT1d_periodic_drive':
			x_volt = x_sample / params_run['sys']['beta']
			x_sample = x_volt


	return t_sample, x_sample, seed_sim, params_run


def analyze_segmentations(
	x_sample,
	full_trace_results,
	params_run,
	parDerived,
	make_plots,
	save_dir,
	analysis_metadata=None,
):

	# Get segmentation parameters
	if analysis_metadata is not None:
		t_win_values = analysis_metadata['t_win_values']
		n_seg_values = analysis_metadata['n_seg_values']
		n_in_win_values = analysis_metadata['n_in_win_values']

	else:
		if params_run['sys']['brownian_motion_type'] == 'OT1d_periodic_drive':
			dt_char = 1 / params_run['sys']['f_drive']
		else:
			dt_char = None
			
		t_win_values, n_seg_values, n_in_win_values = analysis.get_pairs_nseg_twin(
			params_run['ana']['seg_order_min'],
			params_run['ana']['seg_order_max'],
			params_run['ana']['n_seg_tests'],
			params_run['sys']['t_msr'],
			params_run['sys']['dt_sample'],
			dt_char,
			params_run['ana']['create_pairs_nseg_twin_version'],
			params_run['ana']['exclude_seg_smaller'],
		)


	beta_values = np.empty(len(t_win_values))

	gamma_values = np.empty(len(t_win_values))
	kappa_values = np.empty(len(t_win_values))
	d_bead_values = np.empty(len(t_win_values))

	gamma_stds = np.empty(len(t_win_values))
	kappa_stds = np.empty(len(t_win_values))
	d_bead_stds = np.empty(len(t_win_values))

	for it, t_win in enumerate(t_win_values):
		n_seg = n_seg_values[it]
		n_in_win = n_in_win_values[it]

		# Segment trace
		seg_results = analysis.get_segmented_trace(
			x_sample, 
			n_seg, 
			n_in_win
		)

		# FFT
		fft_of_seg_results = analysis.get_fft_of_segments(
			seg_results, 
			params_run['sys']['dt_sample'],
			params_run['sys']['brownian_motion_type'],
			params_run['sys']['f_drive'],
			params_run['ana']['drive_free_segments']
		)

		# Log-bin
		if params_run['ana']['log-binned']:
			if params_run['ana']['log-binning_version'] == 1:
				logbin_results = analysis.get_logbin_of_all_segments_V1(
					fft_of_seg_results,
					bins_per_decade=params_run['ana']['bins_per_decade'],
				)
			elif params_run['ana']['log-binning_version'] == 2:
				logbin_results = analysis.get_logbin_of_all_segments_V2(
					fft_of_seg_results,
					bins_per_decade=params_run['ana']['bins_per_decade'],
				)

			if make_plots:
				# plot the fft of all segments and logbinned results in one plot
				if params_run['plot']['fft_and_logbin_of_segments']:
					plotting.plot_fft_and_logbin_of_segments(
						fft_of_seg_results,
						logbin_results,
						params_run['ana']['fit_mode'],
						n_seg,
						t_win,
						save_dir=save_dir / 'plots/fft_and_logbin/'
					)


		# Fit Lorentzian
		# Decide whether to fit the log-binned or non-log-binned results
		if params_run['ana']['log-binned']:
			to_fit_results = logbin_results
		else:
			to_fit_results = {
				"freq": fft_of_seg_results["freq"].flatten(),
				"PSD": fft_of_seg_results["PSD"].flatten(),
				"std": None,
				"sem": None,
				"n": n_seg,
				"n_freq": fft_of_seg_results["freq"].size,
			}

		fit_results = analysis.get_fitted_parameters(
			to_fit_results,
			params_run['ana']['fit_with_error'],
			params_run['ana']['fit_mode'],
			params_run['ana']['exclude_aliasing_freqs'],
			params_run['ana']['exclude_aliasing_freqs_factor'],
			params_run['ana']['bins_per_decade'],
			params_run['sys']['brownian_motion_type'],
			params_run['sys']['f_drive'],
			params_run['ana']['drive_free_segments']
			)


		# Calibration factor determination
		if not np.isnan(fit_results["fc"]) and not np.isnan(fit_results["var_fc"]):
			if params_run['sys']['brownian_motion_type'] == 'OT1d_periodic_drive':
				if params_run['ana']['drive_free_segments']:
					calibration_results = analysis.get_calibration_factor(
						full_trace_results["freq"],  # full frequency resolution trace
						full_trace_results["PSD_drive_actual"],  # peak of this full resolution
						fit_results,
						params_run['sys']['A_drive'],
						params_run['sys']['f_drive'],
					)
				else:
					calibration_results = analysis.get_calibration_factor(
						to_fit_results["freq"],  # averaged frequency resolution
						fit_results["PSD_drive_actual"],  # peak of averaged frequency resolution
						fit_results,
						params_run['sys']['A_drive'],
						params_run['sys']['f_drive'],
					)
			else:
				calibration_results = {
					'beta': np.nan,
					'A_response': np.nan,
				}
		else:
			calibration_results = {
				'beta': np.nan,
				'A_response': np.nan,
			}


		# Evaluate physical parameters - do calibration
		eval_results = analysis.get_evaluated_parameters(
			fit_results,
			calibration_results,
			params_run['sys']['beta'],
			params_run['phys']['kB'],
			params_run['sys']['T'],
			params_run['sys']['eta'],
			params_run['sys']['brownian_motion_type']
		)

		if make_plots:
			# plot the (non-)log-binned results with the fitted lorentzian
			if np.isnan(fit_results["fc"]) or np.isnan(fit_results["var_fc"]):
				continue  # Skip plotting if fitted parameters contain NaN values

			else:
				if params_run['plot']['lorentzian_fit']:

					PSD_fit = analysis.fit_lorentzian(
						logbin_results["freq"], 
						fit_results["diffusion_constant"], 
						abs(fit_results["fc"])
					)

					plotting.plot_lorentzian_fit(
						PSD_fit,
						to_fit_results,
						fit_results,
						eval_results,
						params_run['ana']['log-binned'],
						parDerived['sys']['gamma'],
						params_run['sys']['kappax'],
						params_run['sys']['d_bead'],
						params_run['sys']['brownian_motion_type'],
						params_run['sys']['f_drive'],
						params_run['sys']['A_drive'],
						params_run['ana']['exclude_aliasing_freqs'],
						params_run['ana']['exclude_aliasing_freqs_factor'],
						params_run['ana']['bins_per_decade'],
						n_seg,
						t_win,
						save_dir=save_dir / 'plots/fits/'
					)

		beta_values[it] = calibration_results['beta']

		gamma_values[it] = eval_results['gamma']
		kappa_values[it] = eval_results['kappa']
		d_bead_values[it] = eval_results['d_bead']

		gamma_stds[it] = eval_results['std_gamma']
		kappa_stds[it] = eval_results['std_kappa']
		d_bead_stds[it] = eval_results['std_d_bead']

	
	# Store results
	segmentation_results = {
		"t_win_values": t_win_values,
		"n_seg_values": n_seg_values,
		"n_in_win_values": n_in_win_values,

		'parameter_means': {
			'beta': beta_values,
			'gamma': gamma_values,
			'kappa': kappa_values,
			'd_bead': d_bead_values
		},
		
		'parameter_stds': {
			'gamma_std': gamma_stds,
			'kappa_std': kappa_stds,
			'd_bead_std': d_bead_stds
		},
	}

	return segmentation_results


def analyze_with_tweezepy(
	x_sample,
	params_run,
	printlog,
):
	"""
	Analyze a sampled trajectory using Tweezepy.
	"""

	if params_run['sys']['brownian_motion_type'] == 'OT1d':

		# Configure tweezepy analysis
		try:
			import tweezepy as tw
		except ImportError:
			printlog("> Warning: 'tweeze' package not found. Please install it to run tweeze analysis.")
			params_run['ana']['use_tweeze_pkg'] = False
			sys.exit()

		tweezepy_results = {
			'par_means_tweeze_psd': None,
			'par_means_tweeze_av': None
		}

		if params_run['use_tweeze']['do_psd']:
			# run tweeze PSD MLE analysis

			# calculate the PSD using the tweezepy package
			psd_tweeze = tw.PSD(x_sample/1e-9, 1/params_run['sys']['dt_sample'])  # it requires the input in nm and Hz, so convert accordingly

			# calculate the fitted parameters using the tweezepy package
			psd_tweeze.mlefit()
			results_psd_tweeze = psd_tweeze.results
			results_psd_tweeze = {
				'g': results_psd_tweeze['g']/1e3,  # Ns/m
				'k': results_psd_tweeze['k']/1e3,  # N/m
			}

			results_psd_tweeze['d_bead'] = 2 * results_psd_tweeze['g'] / (6 * np.pi * params_run['sys']['eta'])

			tweezepy_results['par_means_tweeze_psd'] = {
				'gamma': results_psd_tweeze['g'],
				'kappa': results_psd_tweeze['k'],
				'd_bead': results_psd_tweeze['d_bead']
			}

		if params_run['use_tweeze']['do_AV']:
			# run tweeze Allan Variance analysis

			# calculate the AV using the tweezepy package
			av_tweeze = tw.AV(x_sample/1e-9, 1/params_run['sys']['dt_sample'])  # it requires the input in nm and Hz, so convert accordingly

			# calculate the fitted parameters using the tweezepy package
			av_tweeze.mlefit()
			results_av_tweeze = av_tweeze.results
			results_av_tweeze = {
				'g': results_av_tweeze['g']/1e3,  # Ns/m
				'k': results_av_tweeze['k']/1e3,  # N/m
			}

			results_av_tweeze['d_bead'] = 2 * results_av_tweeze['g'] / (6 * np.pi * params_run['sys']['eta'])

			tweezepy_results['par_means_tweeze_av'] = {
				'gamma': results_av_tweeze['g'],
				'kappa': results_av_tweeze['k'],
				'd_bead': results_av_tweeze['d_bead']
			}

	else:

		# print a warning that the tweeze package only supports the 'OT1d' brownian_motion_type
		printlog("> Warning: 'tweeze' package only supports 'OT1d' brownian_motion_type. Skipping tweeze analysis.")
	print(tweezepy_results)
	return tweezepy_results


def analyze_trace(
	t_sample,
	x_sample,
	seed_sim,
	params_run,
	parDerived,
	make_plots,
	save_dir,
	printlog,
	analysis_metadata=None,
):

	# run actual analysis (?)
	if not params_run['ana']['run_analysis']:
		printlog("Analysis not executed. Stop here.")
		sys.exit()

	start = time.perf_counter()

	# ---------------------------------------------------------
	# Preprocessing
	# ---------------------------------------------------------

	# truncate 
	if params_run['ana']['truncate_trace'] and not params_run['exp']['load_exp_data']:
		if params_run['sys']['brownian_motion_type'] == 'OT1d_periodic_drive':
			t_sample, x_sample, t_msr_trunc = analysis.get_truncated_trace(
				t_sample, 
				x_sample, 
				params_run, 
				'OT1d_periodic_drive',
				printlog
			)

			params_run['sys']['t_msr'] = t_msr_trunc # not ideal, but override

		elif params_run['sys']['brownian_motion_type'] == 'OT1d':
			t_sample, x_sample, t_msr_trunc = analysis.get_truncated_trace(
				t_sample, 
				x_sample, 
				params_run, 
				'OT1d',
				printlog
			)
			params_run['sys']['t_msr'] = t_msr_trunc


	# ---------------------------------------------------------
	# Full trace analysis
	# ---------------------------------------------------------

	full_trace_results = analysis.full_trace_analysis(
		x_sample,
		params_run['sys']['dt_sample'],
		params_run['sys']['f_drive'],
		params_run['sys']['A_drive'],
		params_run['sys']['brownian_motion_type']
	)

	if make_plots:
		# plot the fft and psd of the entire trace
		plotting.plot_fft_and_psd_of_full_trace(
			full_trace_results,
			params_run['sys']['f_drive'],
			params_run['sys']['A_drive'],
			params_run['sys']['brownian_motion_type'],
			save_dir=save_dir / 'plots'
		)

	# ---------------------------------------------------------
	# Segmentation analysis
	# ---------------------------------------------------------
	segmentation_results = analyze_segmentations(
		x_sample,
		full_trace_results,
		params_run,
		parDerived,
		make_plots,
		save_dir,
		analysis_metadata=analysis_metadata,
	)

	analysis_results = {
		"seed": seed_sim,
		"params_run": params_run,
		"parDerived": parDerived,
		**segmentation_results,
	}

	# ---------------------------------------------------------
	# Tweezepy analysis
	# ---------------------------------------------------------
	if params_run['ana']['use_tweeze_pkg']:
		tweezepy_results = analyze_with_tweezepy(
			x_sample,
			params_run,
			printlog
		)

		analysis_results.update(tweezepy_results)

	end = time.perf_counter()
	printlog("  > Elapsed analysis time = {:.2f}s".format((end - start)))

	return analysis_results


def run_analysis(
	base_params, 
	make_plots,
	verbose,
	save_dir,
	analysis_metadata=None,
):

	# set up logging function based on verbosity, to handle printlog calls for larger projects
	printlog = print if verbose else lambda *args, **kwargs: None

	# ---------------------------------------------------------
	# Prepare run / Setup
	# ---------------------------------------------------------
	params_run, parDerived = prepare_run(base_params, printlog)

	# ---------------------------------------------------------
	# Get sampled trace (synthetic or experimental)
	# ---------------------------------------------------------
	t_sample, x_sample, seed_sim, params_run = get_sampled_trace(
		params_run,
		parDerived,
		printlog,
		save_dir,
		analysis_metadata=analysis_metadata,
	)

	# ---------------------------------------------------------
	# Analyze the trace
	# ---------------------------------------------------------
	analysis_results = analyze_trace(
		t_sample,
		x_sample,
		seed_sim,
		params_run,
		parDerived,
		make_plots,
		save_dir,
		printlog,
		analysis_metadata=analysis_metadata,
	)

	return analysis_results





