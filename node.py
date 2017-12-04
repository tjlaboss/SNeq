# Node
#
# Classes and methods for discrete ordinate nodes

import numpy as np

def _cross_sections_from_dict(cross_sections):
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
	sigma_a = 0
	sigma_s = 0
	nu_sigma_f = 0
	
	if "absorption" in cross_sections:
		sigma_a = cross_sections["absorption"]
	if "scatter" in cross_sections:
		sigma_s = cross_sections["scatter"]
	if "nu-fission" in cross_sections:
		nu_sigma_f = cross_sections["fission"]
	sigma_t = sigma_a + sigma_s
	return sigma_s, sigma_a, nu_sigma_f, sigma_t


class Node1D(object):
	"""One-group, 1-D material node with constant properties
	
	Parameters:
	-----------
	dx:                 float, cm; Node width
	quad:               Quadrature; 1D angular quadrature to use
	cross_sections:     dict of {"reaction" : macro_xs}
	source:             float; external source strength.
						[Default: 0]
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
	flux:               scalar flux in the node
	"""
	def __init__(self, dx, quad, cross_sections, source=0.0, name=""):
		self.dx = dx
		self.quad = quad
		self._source = source
		self.name = name
		
		self.sigma_s, self.sigma_a, self.nu_sigma_f, self.sigma_t = \
			_cross_sections_from_dict(cross_sections)
		if self.sigma_t == self.sigma_s:
			self.flux = 0.0
		else:
			self.flux = source/(self.sigma_t - self.sigma_s)
		
		# Precalcuate a commonly-used term
		N = self.quad.N
		self._flux_coeffs = np.empty(N)
		self._flux_denoms = np.empty(N)
		for n in range(N):
			two_mu = 2*self.quad.mus[n]
			prod = dx*self.sigma_t
			self._flux_coeffs[n] = two_mu - prod
			self._flux_denoms[n] = two_mu + prod
	
	def __str__(self):
		rep = """\
Node: {self.name}
	Sigma_s:  {self.sigma_s} cm^-1
	Sigma_a:  {self.sigma_a} cm^-1
	Sigma_t:  {self.sigma_t} cm^-1
	dx:       {self.dx} cm
		""".format(**locals())
		return rep
	
	def _get_source(self):
		"""Get the average nodal source, which is not to be confused with
		the the external source.
		
		NOTE: Does not account for any fission source.
		"""
		qbar = 0.5*(self.sigma_s*self.flux + self._source)
		return qbar
	
	def flux_out(self, flux_in, n):
		"""Calculate the flux leaving the node
		
		Parameters:
		-----------
		flux_in:        float; magnitude of the angular flux entering the node.
						(That's psi[i-1/2] in to get psi[i+1/2] out, or
						 That's psi[i+1/2] in to get psi[i-1/2] out.)
		n:              int; index of the discrete angle
		
		Returns:
		--------
		flux_out:   float; magnitude of the angular flux leaving the node.
		"""
		coeff = self._flux_coeffs[n]
		denom = self._flux_denoms[n]
		qbar = self._get_source()
		flux_out = (flux_in*coeff + 2*self.dx*qbar)/denom
		return flux_out
		
# test
if __name__ == "__main__":
	import quadrature
	sn6 = quadrature.GaussLegendreQuadrature(6)
	nod = Node1D(0.1, sn6, {"scatter": 2.0, "absorption": 0.5}, source=10.0, name="test node")
	fo = nod.flux_out(1.0, 3)
	print(nod)
