# Node
#
# Classes and methods for discrete ordinate nodes

import numpy as np

def _group_cross_sections_from_dict(cross_sections, groups):
	"""Read cross sections from a dictionary.
	
	Parameter:
	----------
	cross_sections:     dict of {"reaction" : macro_xs}
	
	Returns:
	--------
	sigma_a:            float, cm^-1; "absorption" xs
	sigma_s:            float, cm^-1; "scatter" xs
	nu_sigma_f:         float, cm^-1; "nu-fission" xs
	sigma_tr:           float, cm^-1;  transport xs
	"""
	sigma_a = np.zeros(groups)
	sigma_s = np.zeros(groups)
	scatter_matrix = np.zeros((groups, groups))
	nu_sigma_f = np.zeros(groups)
	chi = np.zeros(groups)
	
	if "absorption" in cross_sections:
		sigma_a = cross_sections["absorption"]
	if "scatter" in cross_sections:
		sigma_s = cross_sections["scatter"]
	if "nu-scatter" in cross_sections:
		scatter_matrix = cross_sections["nu-scatter"].squeeze()
	if "nu-fission" in cross_sections:
		nu_sigma_f = cross_sections["nu-fission"]
	if "chi" in cross_sections:
		chi = cross_sections["chi"]
	elif nu_sigma_f.any():
		chi[0] = 1.0
	if "transport" in cross_sections:
		sigma_tr = cross_sections["transport"]
	else:
		#print("Warning: Transport cross section unavailable; using Total.")
		if "total" in cross_sections:
			sigma_tr = cross_sections["total"]
		else:
			sigma_tr = sigma_a + sigma_s
	return sigma_a, sigma_s, scatter_matrix, nu_sigma_f, chi, sigma_tr


class Node(object):
	"""Generic Cartesian node with constant cross sections
	
	Parameters:
	-----------
	dx:                 float, cm; Node width
	quad:               Quadrature; 1D angular quadrature to use
	cross_sections:     dict of {"reaction" : macro_xs}
	source:             float; external source strength.
						[Default: 0]
	groups:             int; number of energy groups
						[Default: 1]
	name:               str; descriptive name for the node
						[Default: empty string]
	
	Attributes:
	-----------
	dx
	quad
	name
	sigma_a
	nu_sigma_f (not implemented yet)
	sigma_s
	sigma_tr
	flux:               scalar flux in the node per energy group
	"""
	def __init__(self, dx, dy, dz, quad, cross_sections, groups, source=0.0, name=""):
		self.dx = dx
		self.dy = dy
		self.dz = dz
		self.source = source
		self._groups = groups
		self.name = name
		
		self.sigma_a, self.sigma_s, self.scatter_matrix,\
		self.nu_sigma_f, self.chi, self.sigma_tr = \
			_group_cross_sections_from_dict(cross_sections, groups)
		
	
	def __str__(self):
		rep = """\
Node: {self.name}
	Sigma_s:  {self.sigma_s} cm^-1
	Sigma_a:  {self.sigma_a} cm^-1
	Sigma_tr: {self.sigma_tr} cm^-1
	nuSigmaF: {self.nu_sigma_f} cm^-1
	dx:       {self.dx} cm
		""".format(**locals())
		return rep


class Node1D(Node):
	"""1-D node with constant cross sections

	Parameters:
	-----------
	dx:                 float, cm; Node width
	quad:               Quadrature; 1D angular quadrature to use
	cross_sections:     dict of {"reaction" : macro_xs}
	source:             float; external source strength.
						[Default: 0]
	groups:             int; number of energy groups
						[Default: 1]
	name:               str; descriptive name for the node
						[Default: empty string]
	"""
	def __init__(self, dx, quad, cross_sections, groups, source=0.0, name=""):
		super().__init__(dx, None, None, quad, cross_sections, groups, source, name)
		# Precalcuate commonly-used terms
		N = quad.N
		self._flux_coeffs = np.empty((N, groups))
		self._flux_denoms = np.empty((N, groups))
		for n in range(N):
			two_mu = 2*abs(quad.mus[n])
			for g in range(groups):
				prod = dx*self.sigma_tr[g]
				self._flux_coeffs[n, g] = two_mu - prod
				self._flux_denoms[n, g] = two_mu + prod
		# TODO: Add optical thickness attributes
		
			

class Node2D(Node):
	"""2-D node with constant cross sections

	Parameters:
	-----------
	dx:                 float, cm; Node width
	dy:                 float, cm; Node depth
	quad:               Quadrature; 2D or 3D angular quadrature to use
	cross_sections:     dict of {"reaction" : macro_xs}
	source:             float; external source strength.
						[Default: 0]
	groups:             int; number of energy groups
						[Default: 1]
	name:               str; descriptive name for the node
						[Default: empty string]
	"""
	def __init__(self, dx, dy, quad, cross_sections, groups, source=0.0, name=""):
		super().__init__(dx, dy, None, quad, cross_sections, groups, source, name)
		# TODO: Add optical thickness attributes
		# TODO: Precalculate flux coefficients