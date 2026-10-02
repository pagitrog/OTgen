import numpy as np
import matplotlib.pyplot as plt
import os


def plot_trace_analysis(
	trace_analysis_results,
	tau,
	dt_sim,
	kBT_over_m,
	save_dir,
):
	os.makedirs(save_dir, exist_ok=True)
	filepath = os.path.join(
		save_dir,
		f'time_scale_analysis.png',
	)

	t_overdamped = trace_analysis_results['t_overdamped']
	x_overdamped = trace_analysis_results['x_overdamped']
	x_inertial = trace_analysis_results['x_inertial']

	l_VCF = trace_analysis_results['l_VCF']
	l_MSD = trace_analysis_results['l_MSD']
	l_VCF_inert = trace_analysis_results['l_VCF_inert']
	l_MSD_inert = trace_analysis_results['l_MSD_inert']

	dt_sim_min = 5*tau # it's about 1% of correlation left, that is exp(-dt_sim/tau) = 1% is about 4.6tau
	
	fig, ax = plt.subplots(1, 3, figsize=(9,3))

	# plot the trace
	ax[0].plot(t_overdamped/tau, x_overdamped/1e-9, color='black', linestyle='dashed')
	ax[0].plot(t_overdamped/tau, x_inertial/1e-9, color='red', linestyle='solid')
	# ax[0,1].set_xlim(ttau_min, ttau_max)
	ax[0].set_xlabel(r"$t/\tau$")
	ax[0].set_ylabel(r"$x[nm]$")
	ax[0].set_box_aspect(1)

	# plot the velocity correlation function (VCF)
	VCF_mean = np.mean(l_VCF, axis=0)
	VCF_inert_mean = np.mean(l_VCF_inert, axis=0)
	VCF_max = VCF_mean[0]
	VCF_inert_max = VCF_inert_mean[0]

	ax[1].plot(
		t_overdamped[:len(VCF_mean)]/tau, 
		VCF_mean/VCF_max, 
		color='black',
		linestyle='dashed',
		label='overdamped'
	)
	ax[1].plot(
		t_overdamped[:len(VCF_inert_mean)]/tau, 
		VCF_inert_mean/VCF_inert_max, 
		color='red',
		linestyle='solid',
		label='inertial'
	)
	ax[1].axvline(
		dt_sim/tau, 
		color='blue', 
		linestyle='dashed', 
		label=r'at $\Delta t_{sim}=%.3g$ [s]' % dt_sim
	)
	ax[1].axvline(
		dt_sim_min/tau, 
		color='green', 
		linestyle='dashed', 
		label=r'at $\approx \Delta t_{sim,min}=5\tau$'
	)
	ax[1].set_xlabel(r"$t/\tau$")
	ax[1].set_ylabel(r"$C_V(t)/C_V(0)$")
	ax[1].set_box_aspect(1)

	ax[1].legend(loc='upper right', fontsize=8)

	# plot the mean squared displacement (MSD)
	MSD_mean = l_MSD
	MSD_inert_mean = l_MSD_inert

	ax[2].loglog(
		t_overdamped[:len(MSD_mean)]/tau, 
		MSD_mean/1e-18, 
		color='black',
		linestyle='dashed'
	)
	ax[2].loglog(
		t_overdamped[:len(MSD_inert_mean)]/tau, 
		MSD_inert_mean/1e-18, 
		color='red',
		linestyle='solid'
	)
	ax[2].axvline(
		dt_sim/tau, 
		color='blue', 
		linestyle='dashed', 
		label=r'at $\Delta t_{sim}=%.3g$ [s]' % dt_sim
	)
	ax[2].axvline(
		dt_sim_min/tau, 
		color='green', 
		linestyle='dashed', 
		label=r'at $\approx \Delta t_{sim,min}=5\tau$'
	)
	ax[2].set_xlabel(r"$t/\tau$")
	ax[2].set_ylabel(r"$\langle x(t)^2 \rangle [\rm nm^2]$")
	ax[2].set_box_aspect(1)

	fig.suptitle(
		r"$\tau = %.3g$ [s]; $\Delta t_{sim} = %.3g$ [s]" % (tau, dt_sim)
	)
	plt.tight_layout()
	plt.savefig(filepath, dpi=200)
	plt.close()


# def plot_segment_lag_correlations(
# 	correlations,
# 	t_win,
# 	dt_sample,
# 	phix,
# 	save_dir
# ):	
# 	os.makedirs(save_dir, exist_ok=True)
# 	filepath = os.path.join(
# 		save_dir,
# 		f'segment_lag_correlations_nseg_{len(correlations)}_twin_{t_win}.png',
# 	)

# 	segment_lag = np.arange(1, len(correlations) + 1)
# 	time_lag = segment_lag * t_win

# 	fig, ax = plt.subplots()

# 	ax.plot(time_lag, correlations, "o-", label="Measured")
# 	ax.axhline(0, color="black", linewidth=1)
# 	ax.plot(
# 		time_lag,
# 		np.exp(-time_lag / phix),
# 		"--",
# 		label=r"$e^{-\Delta t/\phi}$",
# 	)

# 	ax.set_xlabel(r"Segment separation $\Delta t$ [s]")
# 	ax.set_ylabel("Mean Pearson correlation")
# 	ax.legend()
# 	fig.tight_layout()
# 	plt.savefig(filepath, dpi=200)
# 	plt.close()

def plot_fft_and_psd_of_full_trace(
		full_trace_results,
		f_drive,
		A_drive,
		brownian_motion_type,
		save_dir
):	

	os.makedirs(save_dir, exist_ok=True)
	filepath = os.path.join(
		save_dir,
		f'fft_and_psd_of_full_trace.png',
	)

	freq = full_trace_results["freq"]
	PSD = full_trace_results["PSD"]
	f_drive_actual = full_trace_results["f_drive_actual"]
	PSD_drive_actual = full_trace_results["PSD_drive_actual"]

	fig, ax = plt.subplots(1, 1)

	ax.loglog(
		freq, 
		PSD, 
		'.',
		color='tab:blue',
		alpha=0.4,
		label='FFT of full trace'
	)

	# plot the drive frequency and its PSD
	if brownian_motion_type == 'OT1d_periodic_drive':
		ax.axvline(
			f_drive_actual,
			color='red',
			linestyle='dotted',
			label=r'at $f_{drive}, f_{drive,actual}=%.3g, %.3g$ [Hz]' % (f_drive, f_drive_actual)	
		)
		ax.axhline(
			PSD_drive_actual,
			color='green',
			linestyle='dotted',
			label=r'at $P(f_{drive,actual})=%.3g$ [m²/Hz]' % PSD_drive_actual
		)

	ax.set_xlabel('Frequency [1/s]')
	ax.set_ylabel('PSD [m²/1/s]')
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.legend()
	fig.tight_layout()
	plt.savefig(filepath, dpi=200)
	plt.close()


def plot_fft_and_logbin_of_segments(
	fft_of_seg_results,
	logbin_results,
	fit_mode,
	n_seg,
	t_win,
	save_dir
):
	PSD_results = fft_of_seg_results["PSD"]
	freq_results = fft_of_seg_results["freq"]

	logbin_freq = logbin_results["freq"]
	if fit_mode in ('linear', 'log_impr'):
		logbin_PSD = logbin_results["PSD"]
		logbin_yerr = logbin_results["sem"]
	elif fit_mode == 'log':
		PSD_log = logbin_results["PSD_log"]
		sem_log = logbin_results["sem_log"]

		# Convert <log(PSD)> back to linear space
		logbin_PSD = 10**(PSD_log)
		
		# Log-symmetric errors become asymmetric in linear space
		lower_error = logbin_PSD - 10**(PSD_log - sem_log)
		upper_error = 10**(PSD_log + sem_log) - logbin_PSD
		logbin_yerr = np.vstack((lower_error, upper_error))

	os.makedirs(save_dir, exist_ok=True)
	filepath = os.path.join(
		save_dir,
		f'fft_and_logbin_of_segments_nseg_{n_seg}_twin_{t_win}.png',
	)

	fig, ax = plt.subplots(1, 1)

	for i in range(n_seg):
		ax.loglog(
			freq_results[i], 
			PSD_results[i], 
			'.',
			color='tab:blue',
			alpha=0.4)

	ax.errorbar(
		logbin_freq, 
		logbin_PSD, 
		yerr=logbin_yerr,
		fmt='o', 
		alpha=0.5,
		markersize=5,
		color='tab:red', 
		label='log-binned')

	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel('Frequency [1/s]')
	ax.set_ylabel('PSD [m²/1/s]')
	ax.legend()
	fig.tight_layout()
	fig.savefig(filepath)
	plt.close(fig)


def plot_lorentzian_fit(
	PSD_fit,
	logbin_results,
	fit_results,
	eval_results,
	log_binned,
	gamma,
	kappax,
	d_bead,
	brownian_motion_type,
	f_drive,
	A_drive,
	exclude_aliasing_freqs,
	exclude_aliasing_freqs_factor,
	bins_per_decade,
	n_seg,
	t_win,
	save_dir,
):
	os.makedirs(save_dir, exist_ok=True)
	filepath = os.path.join(
		save_dir,
		f"Lorentzian_fit_nseg_{n_seg}_twin_{t_win}s.png",
	)


	# load the data
	freq, PSD = logbin_results["freq"], logbin_results["PSD"]

	diffusion_constant_fit, fc_fit = (
		fit_results["diffusion_constant"],
		abs(fit_results["fc"]),
	)
	std_fc_fit = np.sqrt(fit_results["var_fc"])

	gamma_fit, kappax_fit, d_bead_fit, A_drive_fit = (
		eval_results['gamma'],
		eval_results['kappa'],
		eval_results['d_bead'],
		eval_results['A_response'],
	)
	std_gamma_fit, std_kappax_fit, std_d_bead_fit = (
		eval_results['std_gamma'],
		eval_results['std_kappa'],
		eval_results['std_d_bead']
	)


	# do the plotting
	fig, ax = plt.subplots(1, 1, figsize=(6,4))

	# log-binned data and lorentzian fit
	ax.loglog(
		freq, PSD, 
		'.', alpha=0.4, 
		label=("log_binned " if log_binned else '') + 'FFT'
	)
	ax.loglog(
		freq, PSD_fit, 
		color='black', lw=1, 
		label='Lorentzian fit'
	)


	# plot the fitted corner frequency and the PSD plateau
	ax.axvline(
		fc_fit, 
		color='blue', 
		linestyle='dashed', 
		label=rf'$(f_{{c,fit}}\pm\sigma_{{f_c}})={fc_fit:.3g}\pm$${std_fc_fit:.3g}$ [1/s]'
	)
	PSD_plateau = diffusion_constant_fit / (np.pi**2 * fc_fit**2)
	ax.axhline(
		PSD_plateau, 
		color="red", 
		linestyle='dashed', 
		label=rf'PSD plateau fitted'
	)


	# plot the drive frequency if applicable
	if brownian_motion_type == 'OT1d_periodic_drive':
		ax.axvline(
			f_drive, 
			color='green', 
			linestyle='dashed', 
			label=rf'$f_{{drive}}={f_drive:.3g}$ [1/s]'
		)


	# if exclude_aliasing_freqs is True, plot a vertical line at the frequency where aliasing frequencies start
	if exclude_aliasing_freqs:
		aliasing_freq = freq[-int(exclude_aliasing_freqs_factor*bins_per_decade)]
		ax.axvline(
			aliasing_freq, 
			color='orange', 
			linestyle='dashed', 
			label=rf'exclude $f_{{aliasing}}\geq{aliasing_freq:.3g}$ [1/s]'
		)


	# plot the fitted parameters and their errors
	ax.plot([], [], ' ', 
		label=r"$(\gamma_{soll},\gamma_{fit}\pm\sigma_\gamma)$ = $(%.3g,%.3g\pm$$%.3g)\;1e-9$ [Ns/m]" 
		% (gamma/1e-9, gamma_fit/1e-9, std_gamma_fit/1e-9)
	)
	ax.plot([], [], ' ', 
		label=r"$(\kappa_{soll},\kappa_{fit}\pm\sigma_\kappa)$ = $(%.3g,%.3g\pm$$%.3g)$ [fN/nm]" 
		% (kappax/(1e-15/1e-9), kappax_fit/(1e-15/1e-9), std_kappax_fit/(1e-15/1e-9))
	)
	ax.plot([], [], ' ', 
		label=r"$(d_{soll},d_{fit}\pm\sigma_{d})$ = $(%.4g,%.4g\pm$$%.3g)$ [nm]" 
		% (d_bead/1e-9, d_bead_fit/1e-9, std_d_bead_fit/1e-9)
	)


	# set the labels and legend
	ax.set_xlabel('Frequency $f$ [1/s]')
	ax.set_ylabel('PSD $P(f)$ [m²/1/s]')
	ax.legend(loc='lower left')

	fig.tight_layout()
	fig.savefig(filepath, dpi=200)
	plt.close(fig)



def plot_parameter_comparison_singleTrace(
	analysis_results,
	relative_results,
	save_dir
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
		'rel_kappax': r'$(\kappa_{x,fit}-\kappa_{x,soll})\;/\;\kappa_{x,soll}$',

		'd_bead': r'$d_{bead,fit}$ [m]',
		'abs_d_bead': r'$(d_{bead,fit}-d_{bead,soll})$ [m]',
		'rel_d_bead': r'$(d_{bead,fit}-d_{bead,soll})\;/\;d_{bead,soll}$',
	}

	std_keys = {
		'beta': None,
		'gamma': 'gamma_std',
		'kappa': 'kappa_std',
		'd_bead': 'd_bead_std',
	}

	for res_key, ylabel in ylabels.items():

		# ---------------------------------------------------------
		# Plain fitted values
		# ---------------------------------------------------------
		if res_key in ('beta', 'gamma', 'kappa', 'd_bead'):

			std_key = std_keys[res_key]

			methods = {
				'base': {
					'value': analysis_results['parameter_means'][res_key],
					'std': (
						analysis_results['parameter_stds'][std_key]
						if std_key is not None
						else None
					),
				},
			}

			# Tweezepy
			if res_key != 'beta':

				if analysis_results.get('par_means_tweeze_psd') is not None:
					methods['tweeze_psd'] = {
						'value': analysis_results[
							'par_means_tweeze_psd'
						][res_key],
						'std': None,
					}

				if analysis_results.get('par_means_tweeze_av') is not None:
					methods['tweeze_AV'] = {
						'value': analysis_results[
							'par_means_tweeze_av'
						][res_key],
						'std': None,
					}

		# ---------------------------------------------------------
		# Absolute / relative errors
		# ---------------------------------------------------------
		else:

			methods = {
				'base': {
					'value': relative_results['ana_results'][res_key],
					'std': relative_results[
						'ana_results_std'
					].get(res_key + '_std', None),
				},
			}

			if 'ana_tweeze_results' in relative_results:
				tweeze_results = relative_results[
					'ana_tweeze_results'
				]

				psd_key = res_key + '_psd'
				if psd_key in tweeze_results:
					methods['tweeze_psd'] = {
						'value': tweeze_results[psd_key],
						'std': None,
					}

				av_key = res_key + '_AV'
				if av_key in tweeze_results:
					methods['tweeze_AV'] = {
						'value': tweeze_results[av_key],
						'std': None,
					}

		# ---------------------------------------------------------
		# Plot
		# ---------------------------------------------------------
		plot_parameter_comparison(
			methods,
			analysis_results['params_run']['sys']['t_msr'],
			analysis_results['params_run']['plot'][
				'comparing_parameters_onTop'
			],
			analysis_results['n_seg_values'],
			analysis_results['t_win_values'],
			ylabel,
			res_key,
			save_dir=save_dir / 'plots'
		)


def plot_parameter_comparison(
	methods,
	t_msr,
	plot_t_msr_approx_onTop,
	n_seg_values,
	t_win_values,
	ylabel,
	title,
	save_dir,
):
	os.makedirs(save_dir, exist_ok=True)

	filepath = os.path.join(
		save_dir,
		f'{title}.png',
	)

	colors = (plt.rcParams['axes.prop_cycle'].by_key()['color'])

	markers = {
		'base': 'o',
		'tweeze_psd': 's',
		'tweeze_AV': '^',
	}

	fig, axs = plt.subplots(1, 1)

	# ---------------------------------------------------------
	# Plot analysis methods
	# ---------------------------------------------------------
	for i, (method, results) in enumerate(methods.items()):
		color = colors[i % len(colors)]

		# Base analysis contains one result per t_win
		if method == 'base':

			x_values = np.asarray(t_win_values)
			y_values = np.asarray(results['value'])

			yerr = results['std']
			if yerr is not None:
				yerr = np.asarray(yerr)

		# Tweezepy analyses use the full trace
		elif method in (
			'tweeze_psd',
			'tweeze_AV',
		):

			x_values = np.asarray([t_msr])
			y_values = np.atleast_1d(results['value'])

			yerr = results['std']
			if yerr is not None:
				yerr = np.atleast_1d(yerr)

		else:
			raise ValueError(f"Unknown method: {method}")

		axs.errorbar(
			x_values,
			y_values,
			yerr=yerr,
			fmt=markers.get(method,	'o',),
			markersize=5,
			color=color,
			mec='black',
			ecolor=color,
			elinewidth=1,
			capsize=3,
			zorder=10,
			label=method,
		)

	# ---------------------------------------------------------
	# Mark first and last segmentation
	# ---------------------------------------------------------
	axs.axvline(
		t_win_values[0],
		color='gray',
		linestyle='dashed',
		label=(
			rf'$n_{{seg}}={n_seg_values[0]:.4g}$; '
			rf'$t_{{win}}={t_win_values[0]:.4g}$ s'
		),
	)

	axs.axvline(
		t_win_values[-1],
		color='black',
		linestyle='dashed',
		label=(
			rf'$n_{{seg}}={n_seg_values[-1]:.4g}$; '
			rf'$t_{{win}}={t_win_values[-1]:.4g}$ s'
		),
	)

	# ---------------------------------------------------------
	# Secondary n_seg x-axis
	# ---------------------------------------------------------
	def t_to_n(t_win):
		return t_msr / t_win

	def n_to_t(n_seg):
		with np.errstate(
			divide='ignore',
			invalid='ignore',
		):
			return t_msr / n_seg

	secax = axs.secondary_xaxis(-0.2,functions=(t_to_n,	n_to_t,))

	secax.set_xlabel(r'$n_{seg}$')

	# ---------------------------------------------------------
	# Plot n_seg * t_win
	# ---------------------------------------------------------
	if plot_t_msr_approx_onTop:

		t_tot = (n_seg_values * t_win_values
		)

		axs2 = axs.twinx()
		axs2_color = 'tab:red'

		axs2.plot(
			t_win_values,
			t_tot,
			color=axs2_color,
			marker='x',
			linewidth=0.5,
			zorder=10,
		)

		axs2.axhline(
			t_msr,
			color=axs2_color,
			label=r'$t_{msr}$',
			zorder=0,
		)

		axs2.tick_params(
			axis='y',
			labelcolor=axs2_color,
		)

		axs2.set_ylabel(
			r"$n_{seg}\;t_{win}$ [s]",
			color=axs2_color,
		)

	axs.axhline(
		0.0,
		color='black',
		linewidth=0.8,
		zorder=0,
	)

	axs.set_xscale('log')

	axs.set_ylabel(ylabel)
	axs.set_xlabel(
		r'$t_{win}$ [s]'
	)

	axs.grid(linestyle='dashed')

	axs.legend()

	fig.tight_layout()
	fig.savefig(filepath, dpi=200)
	plt.close(fig)

def plot_simulation_ensemble_method(
	axs,
	x_values,
	data,
	mean,
	std,
	comparing_parameters_type,
	color,
	label,
	marker='o',
):
	"""
	Plot one traces-statistics method.
	"""

	from matplotlib.patches import Patch

	if comparing_parameters_type=='mean-std':
		axs.errorbar(
			x_values,
			mean,
			yerr=std,
			fmt=marker, markersize=5, color=color, mec='black',
			ecolor=color, elinewidth=1, capsize=3,
			zorder=10,
			label=label,
		)

	elif comparing_parameters_type=='box':
		axs.boxplot(
			x=data,
			positions=x_values,
			widths=0.1 * x_values,
			showmeans=True, showfliers=False,
			meanline=True, patch_artist=True,
			boxprops=dict(facecolor=color,color=color,alpha=0.5),
			meanprops=dict(color='green'),
			medianprops=dict(color='red'),
			capprops=dict(color=color),
			whiskerprops=dict(color=color),
			flierprops=dict(markerfacecolor='red', marker='o', markersize=5, linestyle='none', alpha=0.5),
		)

		return Patch(
			facecolor=color,
			edgecolor=color,
			alpha=0.5,
			label=label,
		)

	elif comparing_parameters_type=='violin':
		po = axs.violinplot(
			dataset=data,
			positions=x_values,
			widths=0.15 * x_values,
			showmeans=True,
			showmedians=True,
			showextrema=True,
		)

		# violin bodies
		for body in po['bodies']:
			body.set_facecolor(color)
			body.set_edgecolor(color)
			body.set_alpha(0.5)

		# mean
		po['cmeans'].set_color('green')
		po['cmeans'].set_linestyle('dashed')

		# median
		po['cmedians'].set_color('red')
		po['cmedians'].set_linestyle('solid')

		# extrema
		po['cbars'].set_color(color)
		po['cmins'].set_color(color)
		po['cmaxes'].set_color(color)

		return Patch(
			facecolor=color,
			edgecolor=color,
			alpha=0.5,
			label=label,
		)

	else:
		raise ValueError(
			"Unknown comparing_parameters_type: "
			f"{comparing_parameters_type}"
		)

def plot_ensemble_results(
	N_runs,
	data,
	means,
	stds,
	methods,
	comparing_parameters_type,
	t_msr,
	plot_t_msr_approx_onTop,
	n_seg_values,
	t_win_values,
	ylabel,
	title,
	save_dir,
	x_range_to_plot=(None, None),
	y_range_to_plot=(None, None),
):
	
	os.makedirs(save_dir, exist_ok=True)
	filepath = os.path.join(save_dir, f'{title}.png')

	colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

	markers = {
		'base': 'o',
		'tweeze_psd': 's',
		'tweeze_AV': '^',
	}

	fig, axs = plt.subplots(1, 1)

	legend_handles = []

	from matplotlib.lines import Line2D

	for i, (key, _) in enumerate(methods.items()):
		color = colors[i % len(colors)]

		if key=='base':
			t_win_values_tmp = np.asarray(t_win_values)
		elif key in ('tweeze_psd', 'tweeze_AV'):
			t_win_values_tmp = np.asarray([t_msr]) # according to n_seg=1, on full trace
		else:
			raise ValueError(f"Unknown method: {key}")

		mean_tmp = np.atleast_1d(means[key])
		std_tmp = np.atleast_1d(stds[key])

		data_tmp = np.asarray(data[key])
		if data_tmp.ndim==1:
			data_tmp = data_tmp[:, np.newaxis]

		mask = np.ones(len(t_win_values_tmp), dtype=bool)
		if x_range_to_plot[0] is not None:
			mask &= t_win_values_tmp >= x_range_to_plot[0]
		if x_range_to_plot[1] is not None:
			mask &= t_win_values_tmp <= x_range_to_plot[1]

		t_win_values_tmp = t_win_values_tmp[mask]
		mean_tmp = mean_tmp[mask]
		std_tmp = std_tmp[mask]
		data_tmp = data_tmp[:, mask]

		if len(t_win_values_tmp)==0:
			continue

		handle = plot_simulation_ensemble_method(
			axs=axs,
			x_values=t_win_values_tmp,
			data=data_tmp,
			mean=mean_tmp,
			std=std_tmp,
			comparing_parameters_type=comparing_parameters_type,
			color=color,
			label=key,
			marker=markers[key],
		)

		if handle is not None:
			legend_handles.append(handle)


	# legend
	if comparing_parameters_type=='mean-std':
		axs.legend()

	elif comparing_parameters_type in ('box', 'violin'):
		mean_handle = Line2D([0], [0], color='green', linestyle='dashed', label='Mean')
		median_handle = Line2D([0], [0], color='red', linestyle='solid', label='Median')
		axs.legend(handles=legend_handles + [mean_handle, median_handle])


	# add n_seg as secondary x-axis
	def t_to_n(t_win):
		return t_msr / t_win

	def n_to_t(n_seg):
		with np.errstate(divide='ignore', invalid='ignore'):
			return t_msr / n_seg

	secax = axs.secondary_xaxis(-0.2, functions=(t_to_n, n_to_t))
	secax.set_xlabel(r'$n_{seg}$')


	if plot_t_msr_approx_onTop:
		# add n_seg_values*t_win_values scale
		t_tot = n_seg_values * t_win_values

		mask = np.ones(len(t_win_values), dtype=bool)
		if x_range_to_plot[0] is not None:
			mask &= t_win_values >= x_range_to_plot[0]
		if x_range_to_plot[1] is not None:
			mask &= t_win_values <= x_range_to_plot[1]

		axs2 = axs.twinx()
		axs2_color = 'tab:red'
		axs2.plot(
			t_win_values[mask], 
			t_tot[mask],
			color=axs2_color,
			marker='x',
			linewidth=0.5,
			zorder=5,
		)
		axs2.axhline(
			t_msr,
			color='tab:red',
			label=r'$t_{msr}$',
			zorder=0,
		)

		axs2.tick_params(axis='y', labelcolor=axs2_color)
		axs2.set_ylabel(r"$n_{seg}\;t_{win}$ [s]", color=axs2_color)


	axs.set_xscale('log')

	x_min = x_range_to_plot[0] if x_range_to_plot[0] is not None else 0.8 * np.min(t_win_values)
	x_max = x_range_to_plot[1] if x_range_to_plot[1] is not None else 1.2 * np.max(t_win_values)

	axs.set_xlim(x_min, x_max)
	axs.set_ylim(y_range_to_plot[0], y_range_to_plot[1])

	axs.tick_params(axis='y', labelcolor='black')
	axs.set_ylabel(ylabel)
	axs.set_xlabel(r'$t_{win}$[s]')

	axs.grid(linestyle='dashed')
	axs.set_title(f"over N = {N_runs} runs")

	fig.tight_layout()
	fig.savefig(filepath, dpi=300)
	plt.close()

def plot_compare_ensembles(
	analyses,
	ylabel,
	title,
	save_dir,
	comparing_parameters_type,
	x_range_to_plot=(None, None),
	y_range_to_plot=(None, None),
):
	"""
	Compare traces-statistics results from several analyses.

	Color identifies the different t_msr analyses.

	comparing_parameters_type:
		'mean-std'
		'box'
		'violin'
	"""

	os.makedirs(save_dir, exist_ok=True)
	filepath = os.path.join(save_dir, f'{title}.png')

	colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

	markers = {
		'base': 'o',
		'tweeze_psd': 's',
		'tweeze_AV': '^',
	}

	fig, axs = plt.subplots(1, 1)

	all_x_values = []
	legend_handles = []

	from matplotlib.lines import Line2D

	# Loop over analyses
	for i_analysis, (analysis_label, analysis) in enumerate(analyses.items()):

		color = colors[i_analysis % len(colors)]

		for method, results in analysis['methods'].items():

			# x values
			if method=='base':
				x_values = np.asarray(analysis['t_win'])

			elif method in ('tweeze_psd', 'tweeze_AV'):
				x_values = np.asarray([analysis['t_msr']])

			else:
				raise ValueError(f"Unknown method: {method}")


			mean = np.atleast_1d(results['mean'])
			median = np.atleast_1d(results['median'])
			std = np.atleast_1d(results['std'])

			data = np.asarray(results['data'])

			# Base:
			#	(N_runs, N_t_win)
			#
			# Tweezepy:
			#	(N_runs,) -> (N_runs, 1)

			if data.ndim==1:
				data = data[:, np.newaxis]


			# Restrict t_win range
			mask = np.ones(len(x_values), dtype=bool)

			if x_range_to_plot[0] is not None:
				mask &= x_values >= x_range_to_plot[0]

			if x_range_to_plot[1] is not None:
				mask &= x_values <= x_range_to_plot[1]

			x_values = x_values[mask]
			mean = mean[mask]
			median = median[mask]
			std = std[mask]
			data = data[:, mask]

			if len(x_values)==0:
				continue


			# Overall distance from zero
			mean_abs_mean = np.nanmean(np.abs(mean))
			mean_abs_median = np.nanmean(np.abs(median))

			legend_label = (
				f'{analysis_label}, {method}: '
				r'$\langle|\bar{x}|\rangle$='
				f'{mean_abs_mean:.3g}, '
				r'$\langle|\tilde{x}|\rangle$='
				f'{mean_abs_median:.3g}'
			)


			# Plot
			handle = plot_simulation_ensemble_method(
				axs=axs,
				x_values=x_values,
				data=data,
				mean=mean,
				std=std,
				comparing_parameters_type=comparing_parameters_type,
				color=color,
				label=legend_label,
				marker=markers[method],
			)

			if handle is not None:
				legend_handles.append(handle)

			all_x_values.extend(x_values)


	# Legend
	if comparing_parameters_type=='mean-std':
		axs.legend()

	elif comparing_parameters_type in ('box', 'violin'):
		mean_handle = Line2D([0], [0], color='green', linestyle='dashed', label='Mean')
		median_handle = Line2D([0], [0], color='red', linestyle='solid', label='Median')

		axs.legend(
			handles=legend_handles + [mean_handle, median_handle]
		)

	# Axes
	axs.set_xscale('log')

	all_x_values = np.asarray(all_x_values)

	if all_x_values.size==0:
		raise ValueError(
			"No data remain inside the selected t_win range."
		)

	x_min = (
		x_range_to_plot[0]
		if x_range_to_plot[0] is not None
		else 0.8 * np.min(all_x_values)
	)

	x_max = (
		x_range_to_plot[1]
		if x_range_to_plot[1] is not None
		else 1.2 * np.max(all_x_values)
	)

	axs.set_xlim(x_min, x_max)
	axs.set_ylim(y_range_to_plot[0], y_range_to_plot[1])

	axs.axhline(
		0.0,
		color='black',
		linewidth=0.8,
		zorder=0,
	)

	axs.set_ylabel(ylabel)
	axs.set_xlabel(r'$t_{win}$ [s]')

	axs.grid(
		linestyle='dashed',
		alpha=0.5,
	)

	fig.tight_layout()
	fig.savefig(filepath, dpi=300)
	plt.close(fig)

def plot_exp_data(
	exp_data,
	params_run,
	save_dir
):
	"""
	Plot raw experimental particle and stage data.

	Time traces show a zoomed section of the data.
	Distributions and PSDs are calculated from the complete traces.
	"""
	os.makedirs(save_dir, exist_ok=True)

	dt_sample = params_run['sys']['dt_sample']
	brownian_motion_type = params_run['sys']['brownian_motion_type']

	if brownian_motion_type == 'OT1d_periodic_drive':
		t_plot = 5.0 / params_run['sys']['f_drive']
	else:
		t_plot = min(params_run['sys']['t_msr'], 1.0)

	# particle data
	plot_exp_channels(
		exp_data=exp_data,
		channels=['x', 'y', 'z'],
		dt_sample=dt_sample,
		t_plot=t_plot,
		title=(
			f"Experimental particle data, "
			f"$t_{{msr}}$ = {params_run['sys']['t_msr']:.3g} s, "
			f"$dt_{{sample}}$ = {dt_sample:.3g} s"
		),
		filepath=os.path.join(save_dir, 'raw_exp_data.png'),
		params_run=params_run,
		plot_psd=True,
		psd_filepath=os.path.join(save_dir, 'raw_exp_PSD.png'),
	)

	# stage data
	plot_exp_channels(
		exp_data=exp_data,
		channels=['x_stage', 'y_stage'],
		dt_sample=dt_sample,
		t_plot=t_plot,
		title="Experimental stage data",
		filepath=os.path.join(save_dir, 'raw_exp_stage_data.png'),
		params_run=params_run,
	)


def plot_exp_channels(
	exp_data,
	channels,
	dt_sample,
	t_plot,
	title,
	filepath,
	params_run,
	plot_psd=False,
	psd_filepath=None,
):
	"""
	Plot available experimental channels and their distributions.

	For periodically driven data, overlay the fitted stage motion
	on the selected drive coordinate.

	Optionally plot the PSD of all supplied channels.
	"""

	channels = [
		channel for channel in channels
		if channel in exp_data and len(exp_data[channel]) > 0
	]

	if len(channels) == 0:
		return

	fig, axs = plt.subplots(
		len(channels), 2,
		figsize=(8, 2.5 * len(channels)),
		gridspec_kw={'width_ratios': [2, 1]},
		squeeze=False,
	)

	for i, channel in enumerate(channels):

		data = exp_data[channel]

		label = {
			'x': r'$x$',
			'y': r'$y$',
			'z': r'$z$',
			'x_stage': r'$x_{\mathrm{stage}}$ [$\mu$m]',
			'y_stage': r'$y_{\mathrm{stage}}$ [$\mu$m]',
		}[channel]

		t = np.arange(len(data)) * dt_sample
		mask = t <= t_plot

		# zoomed time trace
		axs[i, 0].plot(
			t[mask],
			data[mask],
			color='black',
			label='data',
		)

		# fitted periodic stage motion
		if (
			params_run['sys']['brownian_motion_type'] == 'OT1d_periodic_drive'
			and channel == f"{params_run['exp']['coordinate_of_drive']}_stage"
		):
			c_drive = params_run['sys']['c_drive']
			A_drive = params_run['sys']['A_drive_fit'] * 1e6
			f_drive = params_run['sys']['f_drive']
			phi_drive = params_run['sys']['phi_drive']

			stage_fit = c_drive + A_drive * np.cos(
				2 * np.pi * f_drive * t[mask] + phi_drive
			)

			A_drive_lab = A_drive * 1e-6 / 1e-9
			axs[i, 0].plot(
				t[mask],
				stage_fit,
				label=(
					r'fit: '
					rf'$f_{{\mathrm{{drive}}}}={f_drive:.3g}$ Hz, '
					rf'$A_{{\mathrm{{drive}}}}={A_drive_lab:.3g}$ nm'
				),
				color='red',
			)
			axs[i, 0].legend()

		axs[i, 0].set_xlim(0, min(t_plot, t[-1]))
		axs[i, 0].set_ylabel(label)

		# distribution of full trace
		axs[i, 1].hist(
			data,
			bins=50,
			orientation='horizontal',
			color='black',
		)

		if i == len(channels) - 1:
			axs[i, 0].set_xlabel('time [s]')
			axs[i, 1].set_xlabel('counts')

	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(filepath, dpi=300)
	plt.close(fig)

	# PSDs of full traces
	if plot_psd:
		fig, axs = plt.subplots(
			len(channels), 1,
			figsize=(8, 2.5 * len(channels)),
			squeeze=False,
		)

		for i, channel in enumerate(channels):

			data = exp_data[channel]
			data_centered = data - np.mean(data)
			n = len(data)

			X = np.fft.rfft(data_centered)
			freq = np.fft.rfftfreq(n, d=dt_sample)
			PSD = (dt_sample / n) * np.abs(X)**2

			# one-sided PSD
			if n % 2 == 0:
				PSD[1:-1] *= 2.0
			else:
				PSD[1:] *= 2.0

			axs[i, 0].loglog(
				freq[1:],
				PSD[1:],
				color='black',
			)

			# drive frequency
			if params_run['sys']['brownian_motion_type'] == 'OT1d_periodic_drive':
				f_drive = params_run['sys']['f_drive']
				axs[i, 0].axvline(
					f_drive,
					color='red',
					linestyle=':',
					label=rf'$f_{{\mathrm{{drive}}}}={f_drive:.3g}$ Hz',
				)
				axs[i, 0].legend()

			axs[i, 0].set_ylabel(
				rf'PSD$_{{{channel}}}$'
			)
			axs[i, 0].grid(alpha=0.2)

		axs[-1, 0].set_xlabel('frequency [Hz]')

		fig.suptitle('Experimental particle PSDs')
		fig.tight_layout()
		fig.savefig(psd_filepath, dpi=300)
		plt.close(fig)


