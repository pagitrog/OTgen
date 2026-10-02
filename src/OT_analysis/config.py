import scipy.constants as const


def create_parameters():
	return {
		'phys': {
			'kB': const.k
		},

		'sys': { # Tolic-Norrelykke et al. (2006) parameters, silica bead in water at 296.2K -  in bulk
			'brownian_motion_type': 'OT1d_periodic_drive',  # {'OT1d', 'OT1d_periodic_drive'}

			't_msr': 60.,  # s; total simulation time	
			'dt_sample': 1e-5,  # s; sampling time for simulation data

			'd_bead': 1.54e-6,  # m; bead diameter
			'm_bead': 4/3 * const.pi * (1.54e-6/2)**3 * 1.05e3,  # kg; bead mass, assuming density of silica
			'T' : 296.2,  # K; temperature
			'eta': 0.9321e-3,  # Ns/m2; viscosity, at T=23.0Celsius
			'kappax': 1e-6,  # N/m; trap-stiffness, from Volpe et al. (2013)

			'f_drive': 16,  # 1/s; periodic drive frequency
			'A_drive' : 208e-9,  # m; periodic drive amplitude
			'beta': 1.0e-6,  # m/V; detector sensitivity
		},

		'sim': {
			'dt_sim': 1e-7,  # s; simulation time-step
			'sample_trajectory': True,  # whether to sample the trajectory or not
			'seed': 25,
			'generate_seed': False,
		},

		'exp': {
			'load_exp_data': False,  # whether to load experimental data or not, of FALSE, generates synthetic data
			'load_exp_data_path': 'input/',  # path to the experimental data file, if applicable
			'trace_i': 0,  # index of the trace to load from the directory
			'data_labels': ['x', 'y', 'z', 'x_stage', 'y_stage'],  
			'coordinate_of_drive': 'y',  # either 'x', 'y', or 'z', should be present as a stage coordinate, e.g. cannot be z if no z_stage
		},

		'ana':{
			# other packages
			'use_tweeze_pkg': True, # note that if True, generating a trace from tweeze has higher priority

			# momentum relaxation time analysis properties
			'do_only_trace_analysis': False,
			'kmax': 10000,  # number of traces to generate for trace analysis
			'dt_res_fac': 1e3,  # factor to reduce the time-resolution of the traces for analysis

			# analysis properties
			'run_analysis': True,
			'truncate_trace': True,  # truncate the driven data to an integer number of periods of the periodic drive, if applicable

			## segmentation properties
			'seg_order_min': 0,
			'seg_order_max': 4,
			'n_seg_tests': 20,  # number of different segmentations to test
			'create_pairs_nseg_twin_version': 4, # enforced versions 3, 4 according to the brownian_motion_type
				# 1: NOT ACCESSIBLE: integer multiple of dt_char (no log pairs, truncates t_msr by truncating t_win)
				# 2: NOT ACCESSIBLE: simple inverse relation (no log pairs, ignores integer multiple for t_win, uses full t_msr)
				# 3: matching t_msr exactly (log pairs, requires truncated t_msr)
				# 4: combining V1 & V3 (log pairs, requires truncated t_msr to be integer multiple of dt_char)
			'create_pairs_version_override': False,  # do NOT set to True, except for you know what you do; enforces using 'create_pairs_nseg_twin_version'
			'exclude_seg_smaller': 0,  # from time to time n_seg<10 fits fail; ==0: all are included
			'drive_free_segments': True, # remove drive frequency from every segment before averaging or log-binning, if applicable; according to the method of Tolic-Norrelykke et al. (2006)

			## frequency log-binning properties
			'log-binned': True,
			'log-binning_version': 2, # 1: mean of flattened segments, 2: mean of every segment
			'bins_per_decade': 15, #

			## fit properties
			'fit_mode': 'log', # 'linear' or 'log_impr' (log<<PSD>>) or 'log' (<<log(PSD)>>)
			'fit_with_error': False,
			'exclude_aliasing_freqs': True,  # it excludes the 'bins_per_decade' last points of the PSD and freq arrays, which are affected by aliasing frequencies, from the fit (see example plots); hard-coded to 1.5 * decade
			'exclude_aliasing_freqs_factor': 0.5,  # factor to multiply bins_per_decade when excluding aliasing frequencies
		},

		'plot': {
			'fft_of_full_trace': True,
			'fft_and_logbin_of_segments': True,
			'lorentzian_fit': True,
			'comparing_parameters': True,
			'comparing_parameters_onTop': False,
		},

		'use_tweeze': {
			'generate_tweeze_trace': False,
			'load_tweeze_trace': False,
			'do_psd': True,
			'do_AV': True
		}
	}



def sets_of_parameters():

	return {
		'Volpe2013': { # Volpe et al. (2013) parameters, for 1um bead in water at 300K
			'r_bead': 1e-6,  # m; bead radius
			'd_bead': 2e-6,  # m; bead diameter
			'm_bead': 1e-15,  # kg; bead mass, assuming density of silica
			'T' : 300,  # K; temperatur
			'eta': 1e-3,  # Ns/m2; viscosity, at T=300K
			'kappax': 1e-6,  # N/m; trap-stiffness, Volpe et al. (2013)	
		},
		'Norrelykke2006a': { # Tolic-Norrelykke et al. (2006) parameters, polystyrene bead in water at 297.6K - near surfaces
			'd_bead': 528e-9,  # m; bead diameter
			'm_bead': 4/3 * const.pi * (528e-9/2)**3 * 1.05e3,  # kg; bead mass, assuming density of polystyrene
			'T' : 297.6,  # K; temperature
			'eta': 0.9024e-3,  # Ns/m2; viscosity, at T=24.4Celsius
			'f_drive': 32,  # 1/s; periodic drive frequency
			'A_drive' : 150e-9,  # m; periodic drive amplitude
			'fc': 2065,  # 1/s; corner-frequency
			'fc_err': 5,  # 1/s; pm error
			'kappax': 1e-15/1e-9,  # N/m; trap-stiffness, from Volpe et al. (2013)
			'f_sample': 65536,  # 1/s; sampling frequency (~15 mu s)
			't_msr': 60.,  # s; total simulation time
			'n_ens': 100,  # number of ensemble members
		},
		'Norrelykke2006b': { # Tolic-Norrelykke et al. (2006) parameters, silica bead in water at 296.2K -  in bulk
			'd_bead': 1.54e-6,  # m; bead diameter
			'm_bead': 4/3 * const.pi * (1.54e-6/2)**3 * 1.05e3,  # kg; bead mass, assuming density of silica
			'T' : 296.2,  # K; temperature
			'eta': 0.9321e-3,  # Ns/m2; viscosity, at T=23.0Celsius
			'kappax': 1e-6,  # N/m; trap-stiffness, from Volpe et al. (2013)
			'f_drive': 28,  # 1/s; periodic drive frequency
			'A_drive' : 208e-9,  # m; periodic drive amplitude
		},
	}