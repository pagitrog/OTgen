import numpy as np
from numba import njit

from . import analysis


@njit
def get_traj_OT1d_overdamped(
	dt_sim,
	dt_sample,
	t_msr,
	x_init,
	noise,
	phix,
	D,
	f_drive,
	A_drive,
	brownian_motion_type,
	sample_trajectory,
):
	"""
	Generate an overdamped 1D optical-trap trajectory.

	If sample_trajectory == True:
		return the trajectory sampled with dt_sample.

	If sample_trajectory == False:
		return the full simulated trajectory with dt_sim.
	"""

	if sample_trajectory:
		n_sample = int(np.ceil(t_msr / dt_sample))

		t = np.empty(n_sample)
		x = np.empty(n_sample)

		# One additional simulation point may be required to
		# interpolate the final sampling point
		n_sim = int(np.ceil(t_msr / dt_sim)) + 1

	else:
		n_sim = int(np.ceil(t_msr / dt_sim))

		t = np.empty(n_sim)
		x = np.empty(n_sim)

	t[0] = 0.0
	x[0] = x_init

	dym_factor = dt_sim / phix
	noise_scale = np.sqrt(2.0 * D * dt_sim)
	omega_drive = 2.0 * np.pi * f_drive

	t_prev = 0.0
	x_prev = x_init

	if sample_trajectory:
		sample_index = 1
		next_sample_time = dt_sample

	for i in range(1, n_sim):
		t_curr = i * dt_sim
		wi = noise[i - 1]

		# add force contributions
		if brownian_motion_type == 'OT1d':
			drive_contribution = 0.0

		elif brownian_motion_type == 'OT1d_periodic_drive':
			v_drive = (
				A_drive * omega_drive
				* np.cos(omega_drive * t_curr)
			)
			drive_contribution = v_drive * dt_sim

		else:
			raise ValueError(
				f"Unknown brownian_motion_type: "
				f"{brownian_motion_type}"
			)

		# thermal noise force
		thermal_noise_contribution = noise_scale * wi

		# update position using overdamped Langevin equation
		x_curr = (
			- dym_factor * x_prev
			+ x_prev
			+ drive_contribution
			+ thermal_noise_contribution
		)

		if sample_trajectory:

			# Save every sampling time lying inside this simulation interval (interpolate if necessary)
			
			while (
				sample_index < n_sample
				and next_sample_time <= t_curr
			):
				fraction = (
					(next_sample_time - t_prev)
					/ (t_curr - t_prev)
				)

				x[sample_index] = (
					x_prev
					+ fraction * (x_curr - x_prev)
				)

				t[sample_index] = next_sample_time

				sample_index += 1
				next_sample_time = (
					sample_index * dt_sample
				)

		else:
			t[i] = t_curr
			x[i] = x_curr

		t_prev = t_curr
		x_prev = x_curr

	return t, x


@njit
def get_traj_OT1d_inertial(
	dt_sim,
	t_msr,
	x_init,
	v_init,
	noise,
	kappa,
	gamma,
	mass,
	kBT,
	f_drive,
	A_drive,
	brownian_motion_type,
):

	n_sim = int(np.ceil(t_msr / dt_sim)) + 1

	t = np.empty(n_sim)
	x = np.empty(n_sim)
	v = np.empty(n_sim)

	# Equilibrium initial state
	t[0] = 0.0
	x[0] = x_init
	v[0] = v_init

	dym_factor_gamma = gamma * dt_sim / mass
	dym_factor_kappa = kappa * dt_sim / mass
	noise_scale = np.sqrt(2.0 * gamma * kBT * dt_sim) / mass
	omega_drive = 2.0 * np.pi * f_drive

	t_prev = 0.0
	x_prev = x_init
	v_prev = v_init

	for i in range(1, n_sim):
		t_curr = i * dt_sim
		wi = noise[i-1]

		if brownian_motion_type == 'OT1d':
			drive_force = 0.0

		elif brownian_motion_type == 'OT1d_periodic_drive':
			# Periodic driving velocity
			v_drive = A_drive * omega_drive * np.cos(omega_drive * t_curr)
			drive_force = gamma * v_drive * dt_sim / mass 

		else:
			raise ValueError(f"Unknown brownian_motion_type: {brownian_motion_type}")

		# thermal noise force
		thermal_noise_force = noise_scale * wi

		# update velocity using inertial Langevin equation
		v_curr = (
			v_prev
			- dym_factor_gamma * v_prev
			- dym_factor_kappa * x_prev
			+ drive_force
			+ thermal_noise_force
		)

		# update position
		x_curr = x_prev + v_curr * dt_sim

		t[i] = t_curr
		x[i] = x_curr
		v[i] = v_curr
		t_prev = t_curr
		x_prev = x_curr
		v_prev = v_curr

	return t, x, v


# @njit
# def get_traj_OT1d_overdamped(
# 	dt_sim, 
# 	t_msr,
# 	x_init,
# 	noise,
# 	phix, 
# 	D, 
# 	gamma,
# 	f_drive,
# 	A,
# 	brownian_motion_type,
# ):
# 	"""
# 	# write a function as get_traj_OT1d, but without sampling, 
# 	# i.e. return the full trajectory with time-step dt_sim, 
# 	# instead of sampled data with time-step dt_sample
# 	"""

# 	n_sim = int(np.ceil((t_msr) / dt_sim))

# 	t = np.empty(n_sim)
# 	x = np.empty(n_sim)

# 	t[0] = 0.0
# 	x[0] = x_init

# 	dym_factor = dt_sim / phix
# 	noise_scale = np.sqrt(2.0 * D * dt_sim)
# 	omega_drive = 2 * np.pi * f_drive

# 	t_prev = 0.0
# 	x_prev = x_init

# 	for i in range(1, n_sim):
# 		t_curr = i * dt_sim
# 		wi = noise[i-1]

# 		# add force contributions
# 		if brownian_motion_type == 'OT1d':
# 			drive_contribution = 0.0

# 		elif brownian_motion_type == 'OT1d_periodic_drive':
# 			v_drive = A * omega_drive * np.cos(omega_drive * t_curr)
# 			drive_contribution = v_drive * dt_sim

# 		else:
# 			raise ValueError(f"Unknown brownian_motion_type: {brownian_motion_type}")

# 		# thermal noise force
# 		thermal_noise_contribution = noise_scale * wi

# 		# update position using overdamped Langevin equation
# 		x_curr = (
# 			- dym_factor * x_prev 
# 			+ x_prev 
# 			+ drive_contribution
# 			+ thermal_noise_contribution
# 		)

# 		t[i] = t_curr
# 		x[i] = x_curr
# 		t_prev = t_curr
# 		x_prev = x_curr

# 	return t, x

# @njit
# def get_traj_OT1d(
# 	dt_sim, 
# 	dt_sample,
# 	t_msr,
# 	x_init, 
# 	phix, 
# 	D, 
# 	seed
# ):
# 	np.random.seed(seed)

# 	# Sampling times: 0, dt_sample, 2*dt_sample, ... < t_msr
# 	n_sample = int(np.ceil((t_msr) / dt_sample))
# 	n_sim = int(np.ceil((t_msr) / dt_sim))

# 	t_sample = np.empty(n_sample)
# 	x_sample = np.empty(n_sample)

# 	t_sample[0] = 0.0
# 	x_sample[0] = x_init

# 	# things to be calculated only once
# 	dym_factor = dt_sim / phix
# 	noise_scale = np.sqrt(2.0 * D * dt_sim)

# 	# step -1
# 	t_prev = 0.0
# 	x_prev = x_init

# 	# run the actual simulation
# 	sample_index = 1
# 	next_sample_time = sample_index * dt_sample  # initial is for sample_index=1

# 	for i in range(1, n_sim+1):  # in steps of simulation dt_sim
# 		# Calculate current simulation run, time and postion
# 		t_curr = i * dt_sim

# 		wi = np.random.normal()
# 		x_curr = -dym_factor * x_prev + x_prev + noise_scale * wi  
		
# 		# Save every sampling time lying inside this simulation interval
# 		while (
# 			sample_index < n_sample  # True, until t_msr reached
# 			and next_sample_time <= t_curr  # False until multiple of dt_sample
# 		):
# 			#  determines where the sampling time lies between the preceding and current simulation points. 
# 			fraction = (
# 				(next_sample_time - t_prev)
# 				/ (t_curr - t_prev)
# 			)  # is 1, if nex_sample_time==t_curr, is 0 if ==t_prev

# 			# store sampled t, x
# 			x_sample[sample_index] = (
# 				x_prev
# 				+ fraction * (x_curr - x_prev)
# 			)  # is x_curr, if fraction==1, is x_curr if fraction==0

# 			t_sample[sample_index] = next_sample_time

# 			sample_index += 1
# 			next_sample_time = sample_index * dt_sample

# 		t_prev = t_curr
# 		x_prev = x_curr
		
# 	return t_sample, x_sample

# @njit
# def get_traj_OT1d_periodic_drive(
	dt_sim, 
	dt_sample,
	t_msr,
	x_init, 
	phix, 
	D, 
	f_drive,
	A,
	seed
# ):
# 	np.random.seed(seed)

# 	# Sampling times: 0, dt_sample, 2*dt_sample, ... < t_msr
# 	n_sample = int(np.ceil((t_msr) / dt_sample))
# 	n_sim = int(np.ceil((t_msr) / dt_sim))

# 	t_sample = np.empty(n_sample)
# 	x_sample = np.empty(n_sample)

# 	t_sample[0] = 0.0
# 	x_sample[0] = x_init

# 	# things to be calculated only once
# 	dym_factor = dt_sim / phix
# 	noise_scale = np.sqrt(2.0 * D * dt_sim)

# 	# step -1
# 	t_prev = 0.0
# 	x_prev = x_init

# 	# run the actual simulation
# 	sample_index = 1
# 	next_sample_time = sample_index * dt_sample  # initial is for sample_index=1

# 	for i in range(1, n_sim+1):  # in steps of simulation dt_sim

# 		# Calculate current simulation run, time and postion
# 		t_curr = i * dt_sim

# 		wi = np.random.normal()
# 		# x_drive = A * np.sin(2*np.pi*f_drive*t_curr)
# 		v_drive = A * (2*np.pi*f_drive) * np.cos(2*np.pi*f_drive*t_curr) * dt_sim
# 		F_T = np.sqrt(2*D*dt_sim)*wi

# 		x_curr = -dym_factor * x_prev + x_prev + v_drive + F_T
		
# 		# Save every sampling time lying inside this simulation interval
# 		while (
# 			sample_index < n_sample  # True, until t_msr reached
# 			and next_sample_time <= t_curr  # False until multiple of dt_sample
# 		):
# 			#  determines where the sampling time lies between the preceding and current simulation points. 
# 			fraction = (
# 				(next_sample_time - t_prev)
# 				/ (t_curr - t_prev)
# 			)  # is 1, if nex_sample_time==t_curr, is 0 if ==t_prev

# 			# store sampled t, x
# 			x_sample[sample_index] = (
# 				x_prev
# 				+ fraction * (x_curr - x_prev)
# 			)  # is x_curr, if fraction==1, is x_curr if fraction==0

# 			t_sample[sample_index] = next_sample_time

# 			sample_index += 1
# 			next_sample_time = sample_index * dt_sample

# 		t_prev = t_curr
# 		x_prev = x_curr
		
# 	return t_sample, x_sample


def get_trajectories_trace_analysis(
		parDerived, 
		params, 
		x_init, 
		v_init, 
		noise
):
	t_overdamped, x_overdamped = get_traj_OT1d_overdamped(
		params['sim']["dt_sim"]/params['ana']['dt_res_fac'], # use dt_sim/1e2 to
		params['sys']['dt_sample'], # use t_msr = dt_sim
		params['sim']['dt_sim'], # use t_msr = dt_sim
		x_init,
		noise,
		parDerived['sys']["phix"],
		parDerived['sys']["D"],
		params['sys']['f_drive'],
		params['sys']['A_drive'],
		params['sys']['brownian_motion_type'],
		False,
	)
	t_inertial, x_inertial, v_inertial = get_traj_OT1d_inertial(
		params['sim']["dt_sim"]/params['ana']['dt_res_fac'], # use dt_sim/1e2 to
		params['sim']['dt_sim'], # use t_msr = dt_sim
		x_init,
		v_init,
		noise,
		params['sys']["kappax"],
		parDerived['sys']["gamma"],
		params['sys']["m_bead"],
		params['phys']['kB'] * params['sys']["T"],
		params['sys']['f_drive'],
		params['sys']['A_drive'],
		params['sys']['brownian_motion_type'],
	)
		
	return {
		't_overdamped': t_overdamped,
		'x_overdamped': x_overdamped,
		't_inertial': t_inertial,
		'x_inertial': x_inertial,
		'v_inertial': v_inertial
	}


def run_trace_analysis(parDerived, params):
	kmax = params['ana']['kmax']

	n_sim = int(np.ceil((params['sim']['dt_sim']) / (params['sim']['dt_sim']/params['ana']['dt_res_fac'])))
	ll_x_overdamped = np.empty((kmax, n_sim))
	ll_x_inertial = np.empty((kmax, n_sim))
	ll_v_inertial = np.empty((kmax, n_sim))
	
	for k in range(kmax):

		x_init = parDerived['sim']['x_init']
		v_init = np.random.normal(scale=np.sqrt(params['phys']['kB'] * params['sys']["T"] / params['sys']["m_bead"]))	
		noise = np.random.normal(size=n_sim)

		# produce a time series with a time-step of dt_sim/1e2, which is much smaller than the simulation time-step dt_sim
		trace_analysis_data = get_trajectories_trace_analysis(
			parDerived, 
			params, 
			x_init, 
			v_init, 
			noise
		)

		ll_x_overdamped[k] = trace_analysis_data['x_overdamped']
		ll_x_inertial[k] = trace_analysis_data['x_inertial'][:len(trace_analysis_data['x_overdamped'])]  # truncate to match the length of x_overdamped
		ll_v_inertial[k] = trace_analysis_data['v_inertial'][:len(trace_analysis_data['x_overdamped'])]  # truncate to match the length of x_overdamped

	l_VCF = analysis.calculate_VCF(ll_x_overdamped, params['sim']['dt_sim']/params['ana']['dt_res_fac'], None)
	l_VCF_inert = analysis.calculate_VCF(ll_x_inertial, params['sim']['dt_sim']/params['ana']['dt_res_fac'], ll_v_inertial)
	l_MSD = analysis.calculate_MSD(ll_x_overdamped)
	l_MSD_inert = analysis.calculate_MSD(ll_x_inertial)

	# align the initial positions
	# x_inertial -= x_inertial[0] - x_overdamped[0]  

	trace_analysis_results = {
		't_overdamped': trace_analysis_data['t_overdamped'], 
		'x_overdamped': trace_analysis_data['x_overdamped'], # just give the last generated one for plotting
		'x_inertial': trace_analysis_data['x_inertial'][:len(trace_analysis_data['x_overdamped'])], # just give the last generated one for plotting
		'l_VCF': l_VCF,
		'l_MSD': l_MSD,
		'l_VCF_inert': l_VCF_inert,
		'l_MSD_inert': l_MSD_inert
	}

	return trace_analysis_results

