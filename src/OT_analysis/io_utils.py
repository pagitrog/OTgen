import os
import pickle
import numpy as np
import pandas as pd
import nptdms
import scipy as sc


def delete_plots_directory(plots_dir):
	if os.path.exists(plots_dir):
		for root, dirs, files in os.walk(plots_dir, topdown=False):
			for name in files:
				os.remove(os.path.join(root, name))
			for name in dirs:
				os.rmdir(os.path.join(root, name))
		os.rmdir(plots_dir)


def save_ensemble_results(
	all_results,
	ensemble_params,
	base_params,
	save_dir,
):
	os.makedirs(save_dir, exist_ok=True)

	# Save complete, undigested results
	with open(os.path.join(save_dir, "all_results.pkl"), "wb",) as file:
		pickle.dump(all_results, file)

	# Save relevant parameter/configuration dictionaries
	with open(os.path.join(save_dir, "ensemble_params.pkl"), "wb",) as file:
		pickle.dump(ensemble_params, file)

	with open(os.path.join(save_dir, "base_params.pkl"), "wb",) as file:
		pickle.dump(base_params, file)

	# Convert results into long-format DataFrame
	rows = []

	for i_run, result in enumerate(all_results):

		n_seg_values = result["n_seg_values"]
		t_win_values = result["t_win_values"]
		n_in_win_values = result["n_in_win_values"]

		# -----------------------------------------------------
		# Base analysis
		# -----------------------------------------------------
		for parameter, values in result["ana_results"].items():
			values = np.atleast_1d(values)

			for i_pair, value in enumerate(values):
				rows.append({
					"run": i_run,
					"method": "base",
					"parameter": parameter,
					"pair_index": i_pair,
					"n_seg": n_seg_values[i_pair],
					"t_win": t_win_values[i_pair],
					"n_in_win": n_in_win_values[i_pair],
					"value": value,
				})

		# -----------------------------------------------------
		# Tweezepy analyses
		# -----------------------------------------------------
		if "ana_tweeze_results" in result:

			for parameter, values in result[
				"ana_tweeze_results"
			].items():

				values = np.atleast_1d(values)

				if parameter.endswith("_psd"):
					method = "tweeze_psd"
				elif parameter.endswith("_AV"):
					method = "tweeze_AV"
				else:
					# Ignore unrelated tweezepy quantities
					continue

				for value in values:
					rows.append({
						"run": i_run,
						"method": method,
						"parameter": parameter,
						"pair_index": np.nan,
						"n_seg": 1,
						"t_win": base_params['sys']['t_msr'],
						"n_in_win": np.nan,
						"value": value,
					})

	ensemble_df = pd.DataFrame(rows)

	ensemble_df.to_csv(
		os.path.join(
			save_dir,
			"ensemble_results.csv",
		),
		index=False,
	)


def load_ensemble_results(load_dir):
	"""
	Load one saved traces-statistics analysis and reconstruct
	the mean/std results required for plotting.
	"""

	with open(os.path.join(load_dir, "all_results.pkl"), "rb",) as file:
		all_results = pickle.load(file)

	with open(os.path.join(load_dir, "base_params.pkl"), "rb",) as file:
		base_params = pickle.load(file)

	N_runs = len(all_results)

	# ---------------------------------------------------------
	# Base analysis
	# ---------------------------------------------------------
	reshaped_results = {
		'base': {
			'rel_gamma': np.stack([
				res['ana_results']['rel_gamma']
				for res in all_results
			]),
			'rel_kappax': np.stack([
				res['ana_results']['rel_kappax']
				for res in all_results
			]),
			'rel_d_bead': np.stack([
				res['ana_results']['rel_d_bead']
				for res in all_results
			]),
			'rel_beta': np.stack([
				res['ana_results']['rel_beta']
				for res in all_results
			]),
		}
	}

	results = {
		'all_results': all_results,
		'base_params': base_params,

		'N_runs': N_runs,
		't_msr': base_params['sys']['t_msr'],

		'segmentation': {
			'n_seg': all_results[0]['n_seg_values'],
			't_win': all_results[0]['t_win_values'],
			'n_in_win': all_results[0]['n_in_win_values'],
		},

		'base': {},
	}
	
	for parameter, values in reshaped_results['base'].items():

		results['base'][parameter] = {
			'mean': np.nanmean(
				values,
				axis=0,
			),
			'std': np.nanstd(
				values,
				axis=0,
				ddof=1,
			),
		}

	# ---------------------------------------------------------
	# Tweeze PSD analysis
	# ---------------------------------------------------------
	if (
		base_params['ana']['use_tweeze_pkg']
		and base_params['sys']['brownian_motion_type'] == 'OT1d'
		and base_params['use_tweeze']['do_psd']
	):

		results['tweeze_psd'] = {}

		tweeze_parameters = {
			'rel_gamma': 'rel_gamma_psd',
			'rel_kappax': 'rel_kappax_psd',
			'rel_d_bead': 'rel_d_bead_psd',
		}

		for parameter, tweeze_parameter in tweeze_parameters.items():

			values = np.stack([
				res['ana_tweeze_results'][tweeze_parameter]
				for res in all_results
			])

			results['tweeze_psd'][parameter] = {
				'mean': np.nanmean(values, axis=0,),
				'std': np.nanstd(values, axis=0, ddof=1,),
			}

	# ---------------------------------------------------------
	# Tweeze AV analysis
	# ---------------------------------------------------------
	if (
		base_params['ana']['use_tweeze_pkg']
		and base_params['sys']['brownian_motion_type'] == 'OT1d'
		and base_params['use_tweeze']['do_AV']
	):

		results['tweeze_AV'] = {}

		tweeze_parameters = {
			'rel_gamma': 'rel_gamma_AV',
			'rel_kappax': 'rel_kappax_AV',
			'rel_d_bead': 'rel_d_bead_AV',
		}

		for parameter, tweeze_parameter in tweeze_parameters.items():

			values = np.stack([
				res['ana_tweeze_results'][tweeze_parameter]
				for res in all_results
			])

			results['tweeze_AV'][parameter] = {
				'mean': np.nanmean(values, axis=0,),
				'std': np.nanstd(values, axis=0, ddof=1,),
			}

	return results


def load_exp_data(load_exp_data_dir, trace_i, data_labels, params_run):
	"""
	Load experimental data from the trace_i-th TDMS file
	found in load_exp_data_dir and extract dt_sample.
	"""

	files = sorted([
		file for file in os.listdir(load_exp_data_dir)
		if file.endswith('.tdms')
	])

	if len(files) == 0:
		raise FileNotFoundError(
			f"No TDMS files found in '{load_exp_data_dir}'."
		)

	if trace_i >= len(files):
		raise IndexError(
			f"trace_i={trace_i}, but only {len(files)} TDMS files were found."
		)

	filepath = os.path.join(load_exp_data_dir, files[trace_i])
	params_run['exp']['load_exp_data_path'] = filepath

	tdms_file = nptdms.TdmsFile.read(filepath)
	group = tdms_file['FPGA2']

	channel_map = {
		'x': 'Ch2',
		'y': 'Ch1',
		'z': 'Ch0',
		'x_stage': 'Ch5',
		'y_stage': 'Ch6',
	}

	unknown_labels = [
		label for label in data_labels
		if label not in channel_map
	]
	if unknown_labels:
		raise ValueError(f"Unknown data labels: {unknown_labels}.")

	exp_data = {}

	for label in data_labels:
		channel = channel_map[label]

		try:
			exp_data[label] = np.array(group[channel][:])

		except KeyError:
			raise ValueError(
				f"Channel '{channel}' corresponding to '{label}' "
				f"not found in TDMS file."
			)

	# sampling interval stored in microseconds -> convert to seconds
	dt_sample = 1e-6 * group['FPGA us'][0]

	if params_run['sys']['dt_sample'] is not None:
		if not np.isclose(params_run['sys']['dt_sample'], dt_sample):
			raise ValueError(
				"Specified 'dt_sample' is inconsistent with experimental data."
			)

	params_run['sys']['dt_sample'] = dt_sample

	return exp_data, params_run

def get_exp_data_and_metadata(params_run):

	exp_data, params_run = load_exp_data(
		params_run['exp']['load_exp_data_dir'],
		params_run['exp']['trace_i'],
		params_run['exp']['data_labels'],
		params_run,
	)
	print("> Loaded experimental data from the file: " + params_run['exp']['load_exp_data_path'])

	exp_data, params_run = get_rest_metadata(exp_data, params_run)
	params_run = get_drive_metadata(exp_data, params_run)

	return exp_data, params_run


def get_rest_metadata(exp_data, params_run):
	"""
	Determine coordinate of drive, measurement time and time array
	from the loaded experimental data.
	"""

	if len(exp_data) == 0:
		raise ValueError("No experimental data available.")

	brownian_motion_type = params_run['sys']['brownian_motion_type']

	if brownian_motion_type == 'OT1d_periodic_drive':
		coordinate = determine_coordinate_of_drive(exp_data, params_run)
		params_run['exp']['coordinate_of_drive'] = coordinate

	elif brownian_motion_type == 'OT1d':
		coordinate = 'x'

	else:
		raise ValueError(
			f"Unknown brownian_motion_type: '{brownian_motion_type}'."
		)

	n_samples = len(exp_data[coordinate])
	dt_sample = params_run['sys']['dt_sample']
	t_msr = n_samples * dt_sample

	if params_run['sys']['t_msr'] is not None:
		if not np.isclose(params_run['sys']['t_msr'], t_msr):
			raise ValueError(
				"Specified 't_msr' is inconsistent with experimental data."
			)

	params_run['sys']['t_msr'] = t_msr
	exp_data['t'] = np.arange(n_samples) * dt_sample

	return exp_data, params_run

def determine_coordinate_of_drive(exp_data, params_run):
	"""
	Determine the driven coordinate from matching particle and
	stage-channel lengths.
	"""

	required = ['x', 'y', 'x_stage', 'y_stage']
	missing = [label for label in required if label not in exp_data]

	if missing:
		raise ValueError(
			f"Required experimental channels missing: {missing}."
		)

	coordinate_user = params_run['exp']['coordinate_of_drive']

	x_matches = len(exp_data['x_stage']) == len(exp_data['x'])
	y_matches = len(exp_data['y_stage']) == len(exp_data['y'])

	if x_matches and not y_matches:
		coordinate = 'x'

	elif y_matches and not x_matches:
		coordinate = 'y'

	elif not x_matches and not y_matches:
		raise ValueError(
			"Neither stage channel has the same length as its "
			"corresponding particle coordinate."
		)

	else:
		if coordinate_user not in ['x', 'y']:
			raise ValueError(
				"Both stage channels could correspond to the drive, but "
				"'coordinate_of_drive' is not set to 'x' or 'y'."
			)

		coordinate = coordinate_user
		print(
			"> Note: Both x_stage and y_stage have suitable lengths. "
			f"Keeping user-specified coordinate_of_drive='{coordinate}'."
		)

	if coordinate != coordinate_user:
		print(
			f"> Note: coordinate_of_drive changed from "
			f"'{coordinate_user}' to '{coordinate}', because only "
			f"{coordinate}_stage has the appropriate length for the "
			f"corresponding particle coordinate."
		)

	return coordinate


def get_drive_metadata(exp_data, params_run):
	"""
	Determine drive frequency and amplitude from the stage trace.
	"""

	if params_run['sys']['brownian_motion_type'] != 'OT1d_periodic_drive':
		return params_run

	coordinate = params_run['exp']['coordinate_of_drive']
	stage = exp_data[f'{coordinate}_stage']
	t = exp_data['t']
	dt_sample = params_run['sys']['dt_sample']

	# Do FFT, PSD estimation, and identify the initial drive frequency
	stage_centered = stage - np.mean(stage)

	X = np.fft.rfft(stage_centered)
	freq = np.fft.rfftfreq(len(stage), d=dt_sample)
	PSD = (dt_sample / len(stage)) * np.abs(X)**2

	i_drive = np.argmax(PSD[1:]) + 1
	f_drive_init = freq[i_drive]

	def drive_model(t, c, A, f, phi):
		return c + A * np.cos(2 * np.pi * f * t + phi)

	c_init = np.mean(stage)
	A_init = np.sqrt(2) * np.std(stage)

	popt, _ = sc.optimize.curve_fit(
		drive_model,
		t,
		stage,
		p0=[c_init, A_init, f_drive_init, 0.0],
	)

	c_drive, A_drive, f_drive_fit, phi_drive = popt

	# Correct the sign of the amplitude and wrap the phase within [-pi, pi]
	if A_drive < 0:
		A_drive = -A_drive
		phi_drive += np.pi
	phi_drive = (phi_drive + np.pi) % (2 * np.pi) - np.pi

	# Set f_drive based on the fitted frequency and the sampling interval
	f_drive_fit = abs(f_drive_fit)
	n_samples_per_period = int(np.round(1.0 / (f_drive_fit * dt_sample)))
	f_drive = 1.0 / (n_samples_per_period * dt_sample)

	# Compare or update A_drive if provided by the user
	A_drive_fit = A_drive * 1e-6
	params_run['sys']['A_drive_fit'] = A_drive_fit
	A_drive_input = params_run['sys']['A_drive']

	if A_drive_input is None:
		params_run['sys']['A_drive'] = A_drive_fit
	else:
		ratio = max(A_drive_input, A_drive_fit) / min(A_drive_input, A_drive_fit)
		if ratio >= 2:
			print(
				f"> Warning: Provided A_drive ({A_drive_input:.3g} m) and "
				f"fitted A_drive ({A_drive_fit:.3g} m) differ by a factor "
				f"of {ratio:.2f}. Keeping the provided A_drive."
			)

	params_run['sys']['f_drive_fit'] = f_drive_fit
	params_run['sys']['f_drive'] = f_drive
	params_run['sys']['phi_drive'] = phi_drive
	params_run['sys']['c_drive'] = c_drive

	return params_run

