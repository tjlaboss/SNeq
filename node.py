# Node
#
# Classes and methods for discrete ordinate nodes
#
# FIXME: Multigroup is simply a placeholder. Energy groups are purely independent and don't do anything.

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
	sigma_t:            float, cm^-1;  total xs
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
	else:
		chi[-1] = 1.0
	if "total" in cross_sections:
		sigma_t = cross_sections["total"]
	else:
		sigma_t = sigma_a + sigma_s
	return sigma_a, sigma_s, scatter_matrix, nu_sigma_f, chi, sigma_t


class Node1D(object):
	"""One-group, 1-D material node with constant properties
	
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
	sigma_t
	flux:               scalar flux in the node per energy group
	"""
	def __init__(self, dx, quad, cross_sections, source=0.0, groups=1, name=""):
		self.dx = dx
		self.quad = quad
		self._source = source
		self._groups = groups
		self.name = name
		
		self.sigma_a, self.sigma_s, self.scatter_matrix,\
		self.nu_sigma_f, self.chi, self.sigma_t = \
			_group_cross_sections_from_dict(cross_sections, groups)
		
		self.flux = np.zeros(groups)
		for g in range(groups):
			if self.sigma_t[g] < self.sigma_s[g]:
				self.flux[g] = source/(self.sigma_t[g] - self.sigma_s[g])
			else:
				self.flux[g] = source
		
		# Precalcuate a commonly-used term
		N = self.quad.N
		self._flux_coeffs = np.empty((N, groups))
		self._flux_denoms = np.empty((N, groups))
		for n in range(N):
			two_mu = 2*abs(self.quad.mus[n])
			for g in range(groups):
				prod = dx*self.sigma_t[g]
				self._flux_coeffs[n, g] = two_mu - prod
				self._flux_denoms[n, g] = two_mu + prod
	
	def __str__(self):
		rep = """\
Node: {self.name}
	Sigma_s:  {self.sigma_s} cm^-1
	Sigma_a:  {self.sigma_a} cm^-1
	Sigma_t:  {self.sigma_t} cm^-1
	dx:       {self.dx} cm
		""".format(**locals())
		return rep
	
	def _get_source(self, g, k):
		"""Get the average nodal source, which is not to be confused with
		the the external source.
		
		Should now include fission source and scattering sources
		
		Parameters:
		-----------
		g:          int; index of the energy group
		k:          float; eigenvalue
		"""
		# If no chi distribution is given, put all of the fission source
		# into the fastest group
		fission = self.get_fission_source(g, k)
		# Evaluate the scatter matrix
		# maybe scattering source is what's screwing up ther problem?
		#scatter = 0
		scatter = (self.scatter_matrix*self.flux)[g]
		'''
		# TODO: Check whether this is correct
		if self.scatter_matrix:
			scatter = (self.scatter_matrix*self.flux)[g]
		else:
			scatter = self.sigma_s[g]*self.flux[g]
		'''
		qbar = 0.5*(scatter + fission + self._source)
		return qbar
	
	def get_fission_source(self, g, k):
		fission = 0.0
		if self.nu_sigma_f:
			for gp in range(self._groups):
				if not self.chi:
					if g == self._groups - 1:
						fission += self.flux[gp]*self.nu_sigma_f[gp]
				else:
					fission += self.chi[g]*self.flux[gp]*self.nu_sigma_f[gp]
			fission /= k
		return fission
	
	def flux_out(self, flux_in, n, g, k=None):
		"""Calculate the flux leaving the node
		
		Parameters:
		-----------
		flux_in:        float; magnitude of the angular flux entering the node.
						(That's psi[i-1/2] in to get psi[i+1/2] out, or
						 That's psi[i+1/2] in to get psi[i-1/2] out.)
		n:              int; index of the discrete angle
		g:              int; index of the energy group
		k:              float; eigenvalue. Required if the node a
						non-zero nu_sigma_f (fission source).
						[Default: None]
		
		Returns:
		--------
		flux_out:   float; magnitude of the angular flux leaving the node.
		"""
		coeff = self._flux_coeffs[n, g]
		denom = self._flux_denoms[n, g]
		qbar = self._get_source(g, k)
		flux_out = (flux_in*coeff + 2*self.dx*qbar)/denom
		return flux_out
		
# test
if __name__ == "__main__":
	import quadrature
	sn6 = quadrature.GaussLegendreQuadrature(6)
	xs = {"scatter": np.array([2.0, 2.1]),
	      "absorption": np.array([0.0, 0.1])}
	nod = Node1D(0.1, sn6, xs, source=10.0, name="test node")
	fo = nod.flux_out(1.0, 3, 0)
	print(nod)
