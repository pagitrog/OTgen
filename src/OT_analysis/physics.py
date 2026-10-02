import numpy as np


def get_tau(m, gamma):
	return m / gamma

def get_gamma(eta, R):
	return 6 * np.pi * eta * R

def get_diffusion_constant(kB, T, gamma):
	return kB * T / gamma

def get_phi(gamma, k):
	return gamma / k

def get_momentum_relaxation_time(m_bead, gamma):
	"""
	Calculate the momentum relaxation time of a bead in a viscous fluid.
	"""
	return m_bead / gamma


def calculate_derived_parameters(params):
	system = params['sys']

	radius_bead = system['d_bead']/2.
	drag_coefficient = get_gamma(system['eta'], radius_bead)  # Ns/m
	diffusion_constant = get_diffusion_constant(
		params['phys']['kB'], 
		system['T'], 
		drag_coefficient
	)  # ; diffusion constant
	dyn_relaxation_time = get_phi(drag_coefficient, system['kappax'])  # s

	x_init = np.random.normal(
		loc=0.0,
		scale=np.sqrt(params['phys']['kB'] * system['T'] / system['kappax'])
	)

	tau = get_momentum_relaxation_time(
		params['sys']['m_bead'],
		drag_coefficient
	)	

	return {
		'sys': {
			'r_bead': radius_bead,
			'gamma': drag_coefficient,
			'D': diffusion_constant,
			'phix': dyn_relaxation_time,
			'tau': tau,
		},
		'sim': {
			'x_init': x_init
		},
	}
