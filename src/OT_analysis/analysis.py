import numpy as np
import sys

import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import pearsonr


def calculate_MSD(ll_x):
	# calculate the MSD by an ensemble average over an array of traces, where each trace is a row in the array

	l_diff = np.diff(ll_x, axis=0)
	l_diff_sq = np.square(l_diff)

	return np.mean(l_diff_sq, axis=0)

def calculate_VCF(ll_x, dt, ll_v=None):
	ll_x = np.asarray(ll_x, dtype=float)
	ll_v = np.asarray(ll_v, dtype=float) if ll_v is not None else None

	# calculate the VCF by an ensemble average over an array of traces, where each trace is a row in the array

	l_VCF = []
	for it, trace in enumerate(ll_x):

		# calculate the velocity time series by finite difference
		if ll_v is not None: # if the velocity time series is provided, use it directly, e.g. the inertial case
			vi = ll_v[it]
		else: # if the velocity time series is not provided, calculate it by finite difference, e.g. the overdamped case
			vi = np.diff(trace) / dt

		# calculate the VCF by an ensemble average over an array of traces, where each trace is a row in the array
		VCF = np.correlate(vi, vi, mode='full')

		N = len(vi)
		# Keep only the second half (as the VCF is symmetric, from -t to +t)
		VCF = VCF[N - 1:] 
		# correct for decreasing number of overlapping pairs
		VCF /= np.arange(N, 0, -1) 

		l_VCF.append(VCF)

	return l_VCF

def get_adjacent_segment_parameter_correlation(param_values):
	"""
	Calculate the Pearson correlation between fitted parameter values
	from neighboring segments.
	Parameters
	----------
	param_values : 1D array
		Fitted parameter for each segment:
		[p_0, p_1, ..., p_{n_seg-1}]
	Returns
	-------
	r : float
		Pearson correlation coefficient between p_i and p_{i+1}.
	p_value : float
		Two-sided p-value for the hypothesis of zero correlation.
	n_pairs : int
		Number of valid neighboring segment pairs used.
	"""

	param_values = np.asarray(param_values, dtype=float)

	x = param_values[:-1]
	y = param_values[1:]

	# Keep only neighboring pairs for which both fits succeeded
	valid = np.isfinite(x) & np.isfinite(y)
	x = x[valid]
	y = y[valid]

	if len(x) < 2:
		return np.nan, np.nan, len(x)
	
	r, p_value = pearsonr(x, y)

	return r, p_value, len(x)


def get_truncated_trace(
	t_sample,
	x_sample,
	params,
	brownian_motion_type,
	printlog,
):
	""" 
	Truncate the time series to a valid measurement length,
	depending on the Brownian-motion type.
	"""
	
	dt_sample = params['sys']['dt_sample']
	t_msr = params['sys']['t_msr']

	if brownian_motion_type == 'OT1d_periodic_drive':
		f_drive = params['sys']['f_drive']
		dt_char = 1.0 / f_drive

		n_periods_float = t_msr * f_drive
		tol = (
			100
			* np.finfo(float).eps
			* max(1.0, abs(n_periods_float))
		)
		n_periods = int(
			np.floor(n_periods_float + tol)
		)
		n_samples_per_period = int(np.round(dt_char / dt_sample))

		if not np.isclose(
			n_samples_per_period * dt_sample,
			dt_char,
			rtol=1e-10,
			atol=1e-15,
		):
			raise ValueError(
				"One drive period must be an integer multiple of dt_sample."
			)
		
		n_samples_trunc = n_periods * n_samples_per_period
		t_msr_trunc = n_samples_trunc * dt_sample

	elif brownian_motion_type == 'OT1d':
		# truncate only to an integer number of sampling intervals
		# take care of floating-point precision issues when checking if n_samples_float is an integer
		n_samples_float = t_msr / dt_sample
		tol = (
			100
			* np.finfo(float).eps
			* max(1.0, abs(n_samples_float))
		)
		n_samples_trunc = int(
			np.floor(n_samples_float + tol)
		)

		t_msr_trunc = n_samples_trunc * dt_sample

	else:
		raise ValueError(
			f"Unknown brownian_motion_type: {brownian_motion_type}"
		)
	
	# check if t_msr and t_msr_trunc are meaningfully different
	if not np.isclose(
		t_msr_trunc,
		t_msr,
		rtol=0.0,
		atol=0.5 * dt_sample,
	):

		t_sample_trunc = t_sample[:n_samples_trunc]
		x_sample_trunc = x_sample[:n_samples_trunc]

		if brownian_motion_type == 'OT1d_periodic_drive':
			printlog(
				f"Warning: Truncated the data to {n_periods} periods "
				f"of the periodic drive, corresponding to a total "
				f"measurement time of {t_msr_trunc:.4f}s, "
				f"instead of the original {t_msr}s."
			)

		elif brownian_motion_type == 'OT1d':
			printlog(
				f"Warning: Truncated the data to an integer number "
				f"of sampling intervals, corresponding to a total "
				f"measurement time of {t_msr_trunc:.4f}s, "
				f"instead of the original {t_msr}s."
			)
			
		return t_sample_trunc, x_sample_trunc, t_msr_trunc

	else:

		return t_sample, x_sample, t_msr

	

	# t_sample_trunc = t_sample[:n_samples_trunc]
	# x_sample_trunc = x_sample[:n_samples_trunc]

	# # Print warning only if truncation is physically meaningful
	# if not np.isclose(
	# 	t_msr_trunc,
	# 	t_msr,
	# 	rtol=0.0,
	# 	atol=0.5 * dt_sample,
	# ):
	# 	if brownian_motion_type == 'OT1d_periodic_drive':
	# 		printlog(
	# 			f"Warning: Truncated the data to {n_periods} periods "
	# 			f"of the periodic drive, corresponding to a total "
	# 			f"measurement time of {t_msr_trunc:.4f}s, "
	# 			f"instead of the original {t_msr}s."
	# 		)

	# 	elif brownian_motion_type == 'OT1d':
	# 		printlog(
	# 			f"Warning: Truncated the data to an integer number "
	# 			f"of sampling intervals, corresponding to a total "
	# 			f"measurement time of {t_msr_trunc:.4f}s, "
	# 			f"instead of the original {t_msr}s."
	# 		)

	# return t_sample_trunc, x_sample_trunc, t_msr_trunc


def truncate_exp_data_to_drive_periods(
	exp_data,
	params_run,
	t_msr_target=None,
):
	"""
	Truncate experimental data to an integer number of complete
	drive periods. If t_msr_target is given, truncate to this
	common measurement time.
	"""

	if params_run['sys']['brownian_motion_type'] != 'OT1d_periodic_drive':
		return exp_data, params_run

	coordinate = params_run['exp']['coordinate_of_drive']

	dt_sample = params_run['sys']['dt_sample']
	f_drive = params_run['sys']['f_drive']

	n_samples_per_period = int(
		np.round(1.0 / (f_drive * dt_sample))
	)

	if not np.isclose(
		n_samples_per_period * dt_sample,
		1.0 / f_drive,
		rtol=1e-5,
	):
		raise ValueError(
			"One drive period is not an integer multiple of dt_sample."
		)

	n_samples_available = len(exp_data[coordinate])

	if t_msr_target is None:
		n_periods = n_samples_available // n_samples_per_period
		n_samples_trunc = n_periods * n_samples_per_period

	else:
		n_samples_trunc = int(
			np.round(t_msr_target / dt_sample)
		)

		if n_samples_trunc > n_samples_available:
			raise ValueError(
				f"Experimental trace is too short: "
				f"{n_samples_available} samples available, "
				f"{n_samples_trunc} required."
			)

		if n_samples_trunc % n_samples_per_period != 0:
			raise ValueError(
				"t_msr_target must contain an integer number "
				"of drive periods."
			)

	stage_label = f'{coordinate}_stage'

	exp_data[coordinate] = exp_data[coordinate][:n_samples_trunc]
	exp_data[stage_label] = exp_data[stage_label][:n_samples_trunc]
	exp_data['t'] = np.arange(n_samples_trunc) * dt_sample

	params_run['sys']['t_msr'] = n_samples_trunc * dt_sample

	return exp_data, params_run


def full_trace_analysis(
	x_sample, 
	dt_sample,
	f_drive,
	A_drive,
	brownian_motion_type
):
	
	# determine the fft and psd of the entire trace
	_, PSD, freq = get_fft_and_psd(
		x_sample,
		dt_sample
	)

	if brownian_motion_type == 'OT1d_periodic_drive':
		# find the actual drive frequency and corresponding PSD value
		i_drive = np.argmin(np.abs(freq - f_drive))
		f_drive_full_trace = freq[i_drive]
		PSD_drive_full_trace = PSD[i_drive]

		return {
			"freq": freq,
			"PSD": PSD,
			"i_drive": i_drive,
			"f_drive_actual": f_drive_full_trace,
			"PSD_drive_actual": PSD_drive_full_trace,
		}
	
	elif brownian_motion_type == 'OT1d':
		return {
			"freq": freq,
			"PSD": PSD,
			"i_drive": None,
			"f_drive_actual": np.nan,
			"PSD_drive_actual": np.nan,
		}


def get_pairs_version_2(
	t_msr,
	n_seg_values,
	dt,
	exclude_seg_smaller
):
	""" Version 2
	takes into account constraints: 
		n_seg * t_win == t_msr
		type(n_seg) = int
	it's principally only t_win = t_msr / n_seg
	"""

	# second get list of t_win
	## t_win will be unique, as n_seg are unique
	t_win_values = t_msr / n_seg_values
	n_in_win_values = np.array(t_win_values / dt).astype(int)
	t_win_values = n_in_win_values * dt  # the actual t_win, correction due to rounding..

	# exclude some values..
	return exclude_pairs(
		t_win_values, 
		n_seg_values, 
		n_in_win_values, 
		exclude_seg_smaller,
		n_in_win_min=6
	)

def get_pairs_version_3(
	t_msr,
	n_seg_target,
	dt,
	exclude_seg_smaller
):
		"""
		Equivalent to version 4, but without the characteristic-time constraint.

		Constraints:
			n_seg * t_win == t_msr
			t_win = n_in_win * dt
			n_seg and n_in_win are integers

		The returned n_seg values are approximately logarithmically spaced.
		"""

		# Total number of sampling intervals in the measurement
		n_total_float = t_msr / dt
		n_total = int(np.round(n_total_float))

		if not np.isclose(
			n_total_float,
			n_total,
			rtol=1e-10,
			atol=1e-12,
		):
			raise ValueError(
				"Exact equality is impossible: t_msr must be an integer "
				"multiple of dt. "
				f"Received t_msr / dt = {n_total_float}."
			)

		# Valid n_seg values must divide the total number of samples exactly
		divisors = np.array(
			[
				n
				for n in range(1, n_total + 1)
				if n_total % n == 0
			],
			dtype=int,
		)

		# Restrict divisors to the requested logarithmic target range
		n_lower = n_seg_target.min()
		n_upper = n_seg_target.max()

		valid_n_seg = divisors[
			(divisors >= n_lower)
			& (divisors <= n_upper)
		]

		if valid_n_seg.size == 0:
			raise ValueError(
				"No exact pairs satisfying all constraints exist "
				"in the requested n_seg range."
			)

		# Select the valid divisor closest to each logarithmic target
		selected = []

		for target in n_seg_target:
			idx = np.argmin(
				np.abs(
					np.log(valid_n_seg)
					- np.log(target)
				)
			)

			selected.append(valid_n_seg[idx])

		n_seg_values = np.unique(selected)

		# Exact construction of window sizes
		n_in_win_values = n_total // n_seg_values
		t_win_values = n_in_win_values * dt

		# Internal consistency checks
		assert np.all(
			n_seg_values * n_in_win_values
			== n_total
		)

		assert np.allclose(
			n_seg_values * t_win_values,
			t_msr,
		)

		return exclude_pairs(
			t_win_values,
			n_seg_values,
			n_in_win_values,
			exclude_seg_smaller,
			n_in_win_min=6,
		)

def get_pairs_version_4(
	t_msr,
	n_seg_target,
	dt,
	dt_char,
	exclude_seg_smaller
):
	"""
	Combines versions 1 and 3.
	Constraints:
		n_seg * t_win == t_msr
		t_win = integer multiple of dt_char
		t_win = n_in_win * dt
		n_seg and n_in_win are integers
	The returned n_seg values are approximately logarithmically spaced.
	"""
	
	# Number of sampling steps per characteristic period
	n_sample_char_float = dt_char / dt
	n_sample_char = int(round(n_sample_char_float))

	if not np.isclose(
		n_sample_char_float,
		n_sample_char,
		rtol=1e-10,
		atol=1e-12,
	):
		raise ValueError(
			"dt_char must be an integer multiple of dt. "
			f"Received dt_char / dt = {n_sample_char_float}."
		)
	
	# Number of characteristic periods in the complete measurement
	n_char_total_float = t_msr / dt_char
	n_char_total = int(round(n_char_total_float))

	if not np.isclose(
		n_char_total_float,
		n_char_total,
		rtol=1e-10,
		atol=1e-12,
	):
		raise ValueError(
			"Exact equality is impossible: t_msr must be an integer "
			"multiple of dt_char. "
			f"Received t_msr / dt_char = {n_char_total_float}."
		)
	
	# Valid n_seg values must divide the total number of
	# characteristic periods exactly
	divisors = np.array(
		[
			n
			for n in range(1, n_char_total + 1)
			if n_char_total % n == 0
		],
		dtype=int,
	)

	# Restrict divisors to the requested logarithmic target range
	n_lower = n_seg_target.min()
	n_upper = n_seg_target.max()
	valid_n_seg = divisors[
		(divisors >= n_lower)
		& (divisors <= n_upper)
	]

	if valid_n_seg.size == 0:
		raise ValueError(
			"No exact pairs satisfying all constraints exist "
			"in the requested n_seg range."
		)
	
	# Select the valid divisor closest to each logarithmic target
	selected = []
	for target in n_seg_target:
		idx = np.argmin(
			np.abs(
				np.log(valid_n_seg)
				- np.log(target)
			)
		)
		selected.append(valid_n_seg[idx])

	n_seg_values = np.unique(selected)
	
	# Exact number of characteristic periods per segment
	n_char_in_win = n_char_total // n_seg_values

	# Exact construction of window sizes
	n_in_win_values = n_char_in_win * n_sample_char
	t_win_values = n_in_win_values * dt

	# Internal consistency checks
	assert np.all(
		n_seg_values * n_char_in_win
		== n_char_total
	)
	assert np.all(
		n_seg_values * n_in_win_values
		== n_char_total * n_sample_char
	)
	assert np.allclose(
		n_seg_values * t_win_values,
		t_msr,
	)
	assert np.allclose(
		t_win_values / dt_char,
		n_char_in_win,
	)
	return exclude_pairs(
		t_win_values,
		n_seg_values,
		n_in_win_values,
		exclude_seg_smaller,
		n_in_win_min=6,
	)

def exclude_pairs(
		t_win, 
		n_seg, 
		n_in_win, 
		val_ex,
		n_in_win_min=6
):
	# exclude pairs where n_seg < val_ex and n_in_win < n_in_win_min
	mask = (n_seg >= val_ex) & (n_in_win >= n_in_win_min)

	return (
		t_win[mask],
		n_seg[mask],
		n_in_win[mask],
	)

def get_pairs_nseg_twin(
		n_min, 
		n_max, 
		n_seg_tests, 
		t_msr, 
		dt, 
		dt_char,
		create_pairs_nseg_twin_version, 
		exclude_seg_smaller,
):
	"""
	Note: it is important that n_in_win = const. for all segments of a given n_seg, as the FFT requires a constant number of points per segment.
	constraints ideally:
		n_seg * t_win == t_msr
		type(n_seg) = int
		type(n_in_win) = int
		type(t_win) = float
		n_seg * t_win = t_msr
		t_win = integer_multiple * dt_char
		t_wina characteristic frequency is satisfied, such as by periodic stage movement
		t_win >> 1 / fc 
		also exclude n_seg < 10, as the fit fails from time to time for small n_seg
	""" 


	# Construct common target values
	n_seg_values = np.round([1, *np.logspace(n_min, n_max, n_seg_tests)]).astype(int)
	n_seg_values = np.asarray(np.unique(n_seg_values), dtype=int)	

	n_seg_target = np.unique(
		np.round([*np.logspace(n_min, n_max, n_seg_tests)]).astype(int)
	)

	# Get version-specific pairs of n_seg and t_win
	if create_pairs_nseg_twin_version==2:
		return get_pairs_version_2(
			t_msr,
			n_seg_values,
			dt,
			exclude_seg_smaller
		)
		

	elif create_pairs_nseg_twin_version==3:
		return get_pairs_version_3(
			t_msr,
			n_seg_target,
			dt,
			exclude_seg_smaller
		)


	elif create_pairs_nseg_twin_version==4:
		return get_pairs_version_4(
			t_msr,
			n_seg_target,
			dt,
			dt_char,
			exclude_seg_smaller
		)

	else:
		raise ValueError(
			f"Unknown pairs_nseg_twin_V: {create_pairs_nseg_twin_version}"
		)


def get_segmented_trace(x_sample, n_seg, n_in_win):
	# Validate that the total number of samples is sufficient
	n_required = n_seg * n_in_win

	if len(x_sample) < n_required:
		raise ValueError(
			f"Insufficient samples: {len(x_sample)} provided, "
			f"but {n_required} required for {n_seg} segments "
			f"of {n_in_win} points each."
		)
	
	return x_sample[:n_required].reshape(n_seg, n_in_win)


def get_fft_and_psd(x_win, dt):
	n_in_win = len(x_win)

	X = np.fft.rfft(x_win)  # actual DFT (discrete fourier trafo)
	PSD = (dt / n_in_win) * np.abs(X)**2
	freq = np.fft.rfftfreq(n_in_win, d=dt)  # DFT sample frequencies (f bin centers)

	return X, PSD[1:], freq[1:]

def get_drive_free_segment(PSD, freq, f_drive):
	# find the actual drive frequency and corresponding PSD value
	i_drive = np.argmin(np.abs(freq - f_drive))

	f_drive_of_segment = freq[i_drive]
	PSD_drive_of_segment = PSD[i_drive]

	# Exclude the drive frequency from the fft results
	mask = np.ones_like(freq, dtype=bool)
	mask[i_drive] = False  # Exclude the drive frequency from the fit
	
	PSD = PSD[mask]
	freq = freq[mask]

	return PSD, freq, PSD_drive_of_segment, f_drive_of_segment, mask

def get_fft_of_segments(
	seg_results, 
	dt_sample,
	brownian_motion_type,
	f_drive,
	drive_free_segments
):
	PSD_results = []
	freq_results = []

	for x_win in seg_results:
		_, PSD, freq = get_fft_and_psd(x_win, dt_sample)

		if (
			brownian_motion_type == 'OT1d_periodic_drive'
			and drive_free_segments
		):
			PSD, freq, PSD_drive_of_segment, f_drive_of_segment, _ = get_drive_free_segment(
				PSD, 
				freq, 
				f_drive
			)

		PSD_results.append(PSD)
		freq_results.append(freq)

	return {
		# "fft": fft_results,
		"PSD": np.asarray(PSD_results),
		"freq": np.asarray(freq_results),
		"std": None
	}


def get_logbin_of_all_segments_V1(fft_of_seg_results, bins_per_decade):
	PSD_results = fft_of_seg_results["PSD"]
	freq_results = fft_of_seg_results["freq"]

	# combine all segments into one array
	PSD_combined = PSD_results.flatten()
	freq_combined = freq_results.flatten()

	# logarithmic binning based on freq, and averaging the PSD values in each bin
	logbin_results = logbin_fft_V1(
		freq_combined, 
		PSD_combined, 
		bins_per_decade=bins_per_decade
	)

	return logbin_results	

def logbin_fft_V1(freq, PSD, bins_per_decade):
	PSD = np.asarray(PSD)

	# All segments should have the same frequency grid (guaranteed by constant n_in_win for every n_seg)
	if np.ndim(freq) == 2:
		freq = np.asarray(freq)[0]
	else:
		freq = np.asarray(freq)

	# Remove non-positive frequencies and corresponding PSD values
	mask = freq > 0
	freq = freq[mask]
	PSD = PSD[mask]

	# Determine the number of logarithmic bins based on the frequency range and bins_per_decade
	n_decades = np.log10(freq.max() / freq.min())
	nbins = int(np.ceil(bins_per_decade * n_decades))

	bin_edges = np.logspace(
		np.log10(freq.min()),
		np.log10(freq.max()),
		nbins + 1, 
		base=10)
	
	bin_indices = np.clip(np.digitize(freq, bin_edges) - 1, 0, nbins - 1) # sort freq into bins, and make sure the indices are within range)

	logbin_freq = []
	logbin_PSD = []
	logbin_std = []  # store the logarithm of the PSD values
	logbin_sem = []
	logbin_PSD_log = []  # store the logarithm of the PSD values
	logbin_std_log = []
	logbin_sem_log = []  
	logbin_n = [] 
	logbin_n_freq = []  

	for i in range(nbins):
		bin_mask = bin_indices == i
		
		if np.any(bin_mask):
			bin_freqs = freq[bin_mask]
			bin_PSDs = PSD[bin_mask]

			# GPT: Use the geometric mean for the representative frequency of a logarithmic bin.
			logbin_freq.append(np.exp(np.mean(np.log(bin_freqs)))) 

			logbin_PSD.append(np.mean(bin_PSDs))
			logbin_std.append(np.std(bin_PSDs))
			logbin_sem.append(np.std(bin_PSDs) / np.sqrt(len(bin_PSDs)))
			
			logbin_PSD_log.append(np.mean(np.log10(bin_PSDs)))
			logbin_std_log.append(np.std(np.log10(bin_PSDs)))			
			logbin_sem_log.append(np.std(np.log10(bin_PSDs)) / np.sqrt(len(bin_PSDs)))

			logbin_n.append(len(bin_PSDs))
			logbin_n_freq.append(np.count_nonzero(bin_mask))

	logbin_results = {
		"freq": np.asarray(logbin_freq),
		"PSD": np.asarray(logbin_PSD),
		"std": np.asarray(logbin_std),
		"sem": np.asarray(logbin_sem),
		"PSD_log": np.asarray(logbin_PSD_log),
		"std_log": np.asarray(logbin_std_log),
		"sem_log": np.asarray(logbin_sem_log),
		"n": np.asarray(logbin_n, dtype=int),
		"n_freq": np.asarray(logbin_n_freq, dtype=int),
	}

	return logbin_results

def get_logbin_of_all_segments_V2(fft_of_seg_results, bins_per_decade):
	PSD_results = np.asarray(fft_of_seg_results["PSD"])
	freq_results = np.asarray(fft_of_seg_results["freq"])

	logbin_results = logbin_fft_V2(
		freq_results,
		PSD_results,
		bins_per_decade=bins_per_decade
	)

	return logbin_results

def logbin_fft_V2(freq, PSD, bins_per_decade):
	freq = np.asarray(freq)
	PSD = np.asarray(PSD)

	# Ensure PSD has shape (n_segments, n_frequencies)
	if PSD.ndim == 1:
		PSD = PSD[np.newaxis, :]

	# All segments have the same frequency grid
	if freq.ndim == 2:
		freq = freq[0]
	if freq.ndim != 1:
		raise ValueError("freq must be a 1D array or a 2D array with one row per segment.")
	
	if PSD.shape[1] != freq.size:
		raise ValueError(
			"The number of PSD frequency points must match the frequency array."
		)
	
	# Remove non-positive frequencies
	mask = freq > 0
	freq = freq[mask]
	PSD = PSD[:, mask]

	if freq.size == 0:
		raise ValueError("No positive frequencies are available for log-binning.")

	# Determine the number of logarithmic bins based on the frequency range and bins_per_decade
	n_decades = np.log10(freq.max() / freq.min())
	nbins = int(np.ceil(bins_per_decade * n_decades))

	bin_edges = np.logspace(
		np.log10(freq.min()),
		np.log10(freq.max()),
		nbins + 1, 
		base=10)
	
	bin_indices = np.clip(np.digitize(freq, bin_edges) - 1, 0, nbins - 1) # sort freq into bins, and make sure the indices are within range)

	logbin_freq = []
	logbin_PSD = []
	logbin_std = []
	logbin_sem = []
	logbin_PSD_log = []  # store the logarithm of the PSD values
	logbin_std_log = []
	logbin_sem_log = []  
	logbin_n = [] # the number of independent segment means that contributed to the final averaged PSD in that bin, given that some segments may not have a valid value in that bin
	logbin_n_freq = [] # the number of original frequency points that were merged into that logarithmic frequency bin

	for i in range(nbins):
		bin_mask = bin_indices == i

		if not np.any(bin_mask):
			continue

		bin_freqs = freq[bin_mask]

		# Mean PSD inside this frequency bin, calculated separately
		segment_means = np.nanmean(PSD[:, bin_mask], axis=1)		

		# takes care of zero values in the PSD that can occur in experimental data
		PSD_bin = PSD[:, bin_mask]
		PSD_bin = np.where(PSD_bin > 0, PSD_bin, np.nan)
		PSD_log_bin = np.nanmean(
			np.log10(PSD_bin),
			axis=1,
		)

		euler_gamma = 0.5772156649015329 # Euler-Mascheroni constant - feel free to ask GPT
		segment_means_log = (
			PSD_log_bin
			+ euler_gamma / np.log(10)
		)
		segment_means = segment_means[np.isfinite(segment_means)]
		segment_means_log = segment_means_log[np.isfinite(segment_means_log)]
		
		n_segments = segment_means.size
		n_segments_log = segment_means_log.size

		if n_segments == 0 or n_segments_log == 0:
			continue

		# Geometric mean as representative logarithmic frequency
		logbin_freq.append(
			np.exp(np.mean(np.log(bin_freqs)))
		)
		logbin_PSD.append(
			np.mean(segment_means)
		)
		logbin_PSD_log.append(
			np.mean(segment_means_log)
		)

		if n_segments > 1:
			std = np.std(segment_means, ddof=1)
			sem = std / np.sqrt(n_segments)
		else:
			# The spread cannot be estimated from only one segment
			std = np.nan
			sem = np.nan

		if n_segments_log > 1:
			std_log = np.std(segment_means_log, ddof=1)
			sem_log = std_log / np.sqrt(n_segments_log)
		else:
			std_log = np.nan
			sem_log = np.nan

		logbin_std.append(std)
		logbin_sem.append(sem)
		logbin_std_log.append(std_log)
		logbin_sem_log.append(sem_log)
		# Number of independent segment averages used for the SEM
		logbin_n.append(n_segments)
		# Number of original frequency points merged into the bin
		logbin_n_freq.append(np.count_nonzero(bin_mask))

	logbin_results = {
		"freq": np.asarray(logbin_freq),
		"PSD": np.asarray(logbin_PSD),
		"std": np.asarray(logbin_std),
		"sem": np.asarray(logbin_sem),
		"PSD_log": np.asarray(logbin_PSD_log),
		"std_log": np.asarray(logbin_std_log),
		"sem_log": np.asarray(logbin_sem_log),
		"n": np.asarray(logbin_n, dtype=int),
		"n_freq": np.asarray(logbin_n_freq, dtype=int),
	}

	return logbin_results


def fit_lorentzian(freq, diffusion_constant, fc):
	return diffusion_constant / (np.pi**2 * (fc**2 + freq**2) )

def fit_lorentzian_log(freq, diffusion_constant, fc):
	return np.log10(diffusion_constant / (np.pi**2 * (fc**2 + freq**2)))


def get_fitted_parameters(
	to_fit_results, 
	fit_with_error, 
	fit_mode,
	exclude_aliasing_freqs,
	exclude_aliasing_freqs_factor,
	bins_per_decade,
	brownian_motion_type,
	f_drive,
	drive_free_segments
):
	freq = to_fit_results["freq"]
	PSD = to_fit_results["PSD"]
	sem = to_fit_results["sem"]

	# Initial guesses for the Lorentzian parameters; use linear PSD 
	A0 = PSD[0] * freq[0]**2 # plateau value of the PSD at low frequencies, estimated from the first point of the PSD
	fc0 = freq[np.argmax(PSD < PSD[0]/2)] 


	# Select the fitting function and data based on the specified fit mode
	if fit_mode=='linear':
		xdata = freq
		ydata = PSD
		sigma = sem
		fit_func = fit_lorentzian
		
	elif fit_mode=='log_impr':  # log<PSD>
		xdata = freq
		ydata = np.log10(PSD)
		sigma = sem / (PSD * np.log(10)) # error propagation for log10
		fit_func = fit_lorentzian_log

	elif fit_mode=='log': # <log(PSD)>
		xdata = freq
		ydata = to_fit_results["PSD_log"]
		sigma = to_fit_results["sem_log"]
		fit_func = fit_lorentzian_log

	# exclude aliasing frequencies from the fit, by removing by the last bins_per_decade points from the data, if requested
	if exclude_aliasing_freqs:
		xdata = xdata[:-int(exclude_aliasing_freqs_factor*bins_per_decade)]
		ydata = ydata[:-int(exclude_aliasing_freqs_factor*bins_per_decade)]
		sigma = sigma[:-int(exclude_aliasing_freqs_factor*bins_per_decade)] if sigma is not None else None

	# If the motion is driven, and the drive frequencies have not been removed from the segments, they will show up in the averaged PSD
	if brownian_motion_type == 'OT1d_periodic_drive':
		if not drive_free_segments: # remove the drive frequency bin from the fit, if not done already for the segments

			ydata, xdata, logPSD_drive_averaged_segments, f_drive_averaged_segments, mask_drive = get_drive_free_segment(ydata, xdata, f_drive)
			sigma = sigma[mask_drive] if sigma is not None else None

			PSD_drive_averaged_segments = np.exp(logPSD_drive_averaged_segments) if fit_mode in ['log_impr', 'log']  else logPSD_drive_averaged_segments
		else:
			PSD_drive_averaged_segments = np.nan
	else:
		PSD_drive_averaged_segments = np.nan


	# Handle the case where there are too few data points to perform a fit
	if len(xdata) < 2:
		return {
			"diffusion_constant": np.nan,
			"fc": np.nan,
			"PSD_drive_actual": np.nan,
			"var_diffusion_constant": np.nan,
			"var_fc": np.nan,
			"cov_fc_diffusion_constant": np.nan,
		}


	# Decide whether to use an error for the fit or not
	if fit_with_error:
		absolute_sigma = True # ! How the sigma parameter affects the estimated covariance depends on absolute_sigma argument, as described above.
	else:
		sigma = None
		absolute_sigma = False


	# Perform the curve fitting with error handling for potential issues
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", OptimizeWarning)

		try:
			popt, pcov = curve_fit(
				fit_func,
				xdata,
				ydata,
				sigma=sigma,
				absolute_sigma=absolute_sigma, # ! How the sigma parameter affects the estimated covariance depends on absolute_sigma argument, as described above.
				p0=[A0, fc0],
				maxfev=10000
			)

			diffusion_constant_fit = popt[0]
			fc_fit = popt[1]


			fit_results = {
				"diffusion_constant": diffusion_constant_fit,
				"fc": fc_fit,
				"PSD_drive_actual": PSD_drive_averaged_segments,
				"var_diffusion_constant": pcov[0, 0],
				"var_fc": pcov[1, 1],
				"cov_fc_diffusion_constant": pcov[1, 0],
			}

			# handle the case that any of values in fit_results are infinity, by setting all values to nan
			if any(np.isinf(list(fit_results.values()))):
				fit_results = {
					"diffusion_constant": np.nan,
					"fc": np.nan,
					"PSD_drive_actual": np.nan,
					"var_diffusion_constant": np.nan,
					"var_fc": np.nan,
					"cov_fc_diffusion_constant": np.nan
				}

		# handle the case that the fit fails, by setting all values to nan
		except (RuntimeError, ValueError):
			fit_results = {
				"diffusion_constant": np.nan,
				"fc": np.nan,
				"PSD_drive_actual": np.nan,
				"var_diffusion_constant": np.nan,
				"var_fc": np.nan,
				"cov_fc_diffusion_constant": np.nan
			}

	return fit_results

def get_calibration_factor(
	freq,
	PSD_drive_actual,
	fit_results,
	A_drive,
	f_drive,
):
	"""
	Determine beta from the driven PSD peak.
	Assumes:
		x = beta * x_volt
	and an undoubled positive-frequency rFFT PSD.
	"""

	fc_fit = abs(fit_results["fc"])
	diffusion_constant_fit = fit_results["diffusion_constant"]

	df = freq[1] - freq[0]

	# Thermal PSD background at the drive frequency,
	## always evaluated in linear PSD space -  get the theoretical value of the thermal background PSD at f_drive
	PSD_background = fit_lorentzian(
		f_drive,
		diffusion_constant_fit,
		fc_fit,
	)

	## Power in the drive spike from the undoubled rFFT PSD: calculated as the area under the spike, minus the thermal background
	W_drive_raw = max(
		(PSD_drive_actual - PSD_background) * df,
		0.0,
	)

	## Convert to the one-sided peak power used in the paper
	W_ex = 2.0 * W_drive_raw

	# Theoretical response power in physical units [m²]
	W_th = (
		A_drive**2/ ( 2.0 * (1.0 + (fc_fit / f_drive)**2)
		)
	)

	if W_ex <= 0.0:
		beta = np.nan
		A_response = np.nan
	else:
		beta = np.sqrt(W_th / W_ex)
		# Measured bead-response amplitude in detector units
		A_response = 2.0 * np.sqrt(W_drive_raw)

	return {
		"beta": beta,
		"A_response": A_response
	}


def get_gamma_fit(
	diffusion_constant_fit, 
	kB, 
	T
):
	return kB * T / (2 * diffusion_constant_fit)

def get_kappa_fit(
	diffusion_constant_fit, 
	fc_fit, 
	kB, 
	T
):
	gamma_fit = get_gamma_fit(diffusion_constant_fit, kB, T)
	return fc_fit * 2 * np.pi * gamma_fit

def get_diameter_fit(
	diffusion_constant_fit, 
	kB, 
	T, 
	eta
):
	gamma_fit = get_gamma_fit(diffusion_constant_fit, kB, T)
	return 2 * gamma_fit / (6 * np.pi * eta)

def get_std_gamma_fit(
	diffusion_constant_fit, 
	gamma_fit, 
	std_diffusion_constant_fit
):
	return gamma_fit / diffusion_constant_fit * std_diffusion_constant_fit

def get_std_kappa_fit(
	diffusion_constant_fit, 
	fc_fit, 
	kappa_fit, 
	std_diffusion_constant_fit, 
	std_fc_fit, 
	cov_fc_diffusion_constant_fit
):
	contribution_noncorrelated = (
		(kappa_fit / diffusion_constant_fit)**2 * std_diffusion_constant_fit**2 
		+ (kappa_fit / fc_fit)**2 * std_fc_fit**2
	)
	contribution_correlated = (
		- 2 * (kappa_fit**2) / (diffusion_constant_fit * fc_fit) * cov_fc_diffusion_constant_fit
	)
	return np.sqrt(contribution_noncorrelated + contribution_correlated)

def get_std_diameter_fit(
	diffusion_constant_fit, 
	d_bead_fit, 
	std_diffusion_constant_fit
):
	return d_bead_fit / diffusion_constant_fit * std_diffusion_constant_fit


def get_evaluated_parameters(
	fit_results,
	calibration_results, 
	beta_provided,
	kB,
	T, 
	eta,
	brownian_motion_type
):
	
	if np.isnan(fit_results["fc"]) or np.isnan(fit_results["var_fc"]):
		return {
			"gamma": np.nan,
			"kappa": np.nan,
			"d_bead": np.nan,
			"A_response": np.nan,
			"std_gamma": np.nan,
			"std_kappa": np.nan,
			"std_d_bead": np.nan
		}
	else:
		fc_fit = abs(fit_results["fc"])
		diffusion_constant_fit = fit_results["diffusion_constant"]
		std_diffusion_constant_fit = np.sqrt(fit_results["var_diffusion_constant"])
		std_fc_fit = np.sqrt(fit_results["var_fc"])
		cov_fc_diffusion_constant_fit = fit_results["cov_fc_diffusion_constant"]

		beta_fit = calibration_results["beta"]
		A_response_fit = calibration_results["A_response"]


	# Calibrate with beta: either beta_fit, or beta_provided, or do nothing
	if brownian_motion_type == 'OT1d_periodic_drive':
		if np.isnan(beta_fit): # only for periodic data, a calibration is attempted, and can thus fail
			return {
				"gamma": np.nan,
				"kappa": np.nan,
				"d_bead": np.nan,
				"A_response": np.nan,
				"std_gamma": np.nan,
				"std_kappa": np.nan,
				"std_d_bead": np.nan
			}
		else: # if calibration succeded
			diffusion_constant_fit = diffusion_constant_fit * beta_fit**2
			std_diffusion_constant_fit = std_diffusion_constant_fit * beta_fit**2
			cov_fc_diffusion_constant_fit = cov_fc_diffusion_constant_fit * beta_fit**2
			
	else: # in the case of non-driven data
		if beta_provided is np.nan: # assume data is already calibrated, use unmodified diff_const, etc.
			pass
		else: # use provided beta for calibration 
			diffusion_constant_fit = diffusion_constant_fit * beta_provided**2
			std_diffusion_constant_fit = std_diffusion_constant_fit * beta_provided**2
			cov_fc_diffusion_constant_fit = cov_fc_diffusion_constant_fit * beta_provided**2


	# calculate derived paramters
	gamma_fit = get_gamma_fit(
		diffusion_constant_fit, 
		kB, 
		T
	)
	kappa_fit = get_kappa_fit(
		diffusion_constant_fit, 
		fc_fit, 
		kB, 
		T
	)
	d_bead_fit = get_diameter_fit(
		diffusion_constant_fit,
		kB,
		T,
		eta,
	)

	A_response_fit = A_response_fit * beta_fit  # include the factor beta in the response amplitude, if the motion is driven

	# calculate std of derived parameters
	std_gamma_fit = get_std_gamma_fit(
		diffusion_constant_fit,
		gamma_fit,
		std_diffusion_constant_fit
	)	
	std_kappa_fit = get_std_kappa_fit(
		diffusion_constant_fit,
		fc_fit,
		kappa_fit,
		std_diffusion_constant_fit,
		std_fc_fit,
		cov_fc_diffusion_constant_fit
	)
	std_d_bead_fit = get_std_diameter_fit(
		diffusion_constant_fit,
		d_bead_fit,
		std_diffusion_constant_fit
	)

	return {
		"gamma": gamma_fit,
		"kappa": kappa_fit,
		"d_bead": d_bead_fit,
		"A_response": A_response_fit,
		"std_gamma": std_gamma_fit,
		"std_kappa": std_kappa_fit,
		"std_d_bead": std_d_bead_fit,
	}



def get_relative_error(fitted_mean_value, true_value):
	# Compare the fitted mean value with the true value and return the absolute and relative differences.
	abs_value = fitted_mean_value - true_value
	rel_value = abs_value / true_value

	return abs_value, rel_value


def get_relative_parameters(analysis_results):

	# calculate absolute and relative fitted parameters
	beta_abs, beta_rel = get_relative_error(
		analysis_results['parameter_means']['beta'],
		analysis_results['params_run']['sys']['beta']
	)
	gamma_abs, gamma_rel = get_relative_error(
		analysis_results['parameter_means']['gamma'], 
		analysis_results['parDerived']['sys']['gamma']
	)
	kappax_abs, kappax_rel = get_relative_error(
		analysis_results['parameter_means']['kappa'], 
		analysis_results['params_run']['sys']['kappax']
	)
	d_bead_abs, d_bead_rel = get_relative_error(
		analysis_results['parameter_means']['d_bead'], 
		analysis_results['params_run']['sys']['d_bead']
	)

	ana_results = {
		'abs_beta': beta_abs,
		'rel_beta': beta_rel,
		'abs_gamma': gamma_abs, 
		'rel_gamma': gamma_rel,
		'abs_kappax': kappax_abs, 
		'rel_kappax': kappax_rel,
		'abs_d_bead': d_bead_abs, 
		'rel_d_bead': d_bead_rel,
	}

	ana_results_std = {
		'abs_beta_std': None,
		'rel_beta_std': None,
		'abs_gamma_std': analysis_results['parameter_stds']['gamma_std'], # its linear for abs and rel, so the std is the same for both
		'rel_gamma_std': None,
		'abs_kappax_std': analysis_results['parameter_stds']['kappa_std'],
		'rel_kappax_std': None,
		'abs_d_bead_std': analysis_results['parameter_stds']['d_bead_std'],
		'rel_d_bead_std': None,
	}

	relative_results = {
		'ana_results': ana_results,
		'ana_results_std': ana_results_std,
	}

	if analysis_results['params_run']['ana']['use_tweeze_pkg']:
		if analysis_results['params_run']['sys']['brownian_motion_type'] == 'OT1d':
			if analysis_results['params_run']['use_tweeze']['do_psd']:

				gamma_abs_tw_psd, gamma_rel_tw_psd = get_relative_error(
					analysis_results['par_means_tweeze_psd']['gamma'], 
					analysis_results['parDerived']['sys']['gamma']
				)
				kappax_abs_tw_psd, kappax_rel_tw_psd = get_relative_error(
					analysis_results['par_means_tweeze_psd']['kappa'], 
					analysis_results['params_run']['sys']['kappax']
				)
				d_bead_abs_tw_psd, d_bead_rel_tw_psd = get_relative_error(
					analysis_results['par_means_tweeze_psd']['d_bead'], 
					analysis_results['params_run']['sys']['d_bead']
				)

				ana_tweeze_results = {
					'abs_gamma_psd': gamma_abs_tw_psd,
					'rel_gamma_psd': gamma_rel_tw_psd,
					'abs_kappax_psd': kappax_abs_tw_psd,
					'rel_kappax_psd': kappax_rel_tw_psd,
					'abs_d_bead_psd': d_bead_abs_tw_psd,
					'rel_d_bead_psd': d_bead_rel_tw_psd,
				}

				relative_results['ana_tweeze_results'] = ana_tweeze_results

			if analysis_results['params_run']['use_tweeze']['do_AV']:

				gamma_abs_tw_av, gamma_rel_tw_av = get_relative_error(
					analysis_results['par_means_tweeze_av']['gamma'], 
					analysis_results['parDerived']['sys']['gamma']
				)
				kappax_abs_tw_av, kappax_rel_tw_av = get_relative_error(
					analysis_results['par_means_tweeze_av']['kappa'], 
					analysis_results['params_run']['sys']['kappax']
				)
				d_bead_abs_tw_av, d_bead_rel_tw_av = get_relative_error(
					analysis_results['par_means_tweeze_av']['d_bead'], 
					analysis_results['params_run']['sys']['d_bead']
				)

				ana_tweeze_results = {
					'abs_gamma_AV': gamma_abs_tw_av,
					'rel_gamma_AV': gamma_rel_tw_av,
					'abs_kappax_AV': kappax_abs_tw_av,
					'rel_kappax_AV': kappax_rel_tw_av,
					'abs_d_bead_AV': d_bead_abs_tw_av,
					'rel_d_bead_AV': d_bead_rel_tw_av,
				}

				if 'ana_tweeze_results' in relative_results:
					relative_results['ana_tweeze_results'].update(ana_tweeze_results)
				else:
					relative_results['ana_tweeze_results'] = ana_tweeze_results

	return relative_results
