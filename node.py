# Node
#
# Classes and methods for discrete ordinate nodes

import numpy as np


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
	sigma_tr
	flux:               scalar flux in the node per energy group
	"""
	def __init__(self, dx, quad, material, source=0.0, name=""):
		self.dx = dx
		self.quad = quad
		self._source = source
		self.name = name
		
		groups = material.groups
		self.sigma_a = material.sigma_a
		self.scatter_matrix = material.scatter_matrix
		self.nu_sigma_f = material.nu_sigma_f
		self.chi = material.chi
		self.sigma_tr = material.sigma_tr
		self.D = material.D
		self.sigma_r = material.sigma_r
		
		
		# Precalcuate a commonly-used term
		N = self.quad.N
		self._flux_coeffs = np.empty((N, groups))
		self._flux_denoms = np.empty((N, groups))
		for n in range(N):
			two_mu = 2*abs(self.quad.mus[n])
			for g in range(groups):
				prod = dx*self.sigma_tr[g]
				self._flux_coeffs[n, g] = two_mu - prod
				self._flux_denoms[n, g] = two_mu + prod
	
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


# test
if __name__ == "__main__":
	import quadrature
	sn6 = quadrature.GaussLegendreQuadrature(6)
	xs = {"scatter": np.array([2.0, 2.1]),
	      "absorption": np.array([0.0, 0.1])}
	nod = Node1D(0.1, sn6, xs, source=10.0, name="test node")
	fo = nod.flux_out(1.0, 3, 0)
	print(nod)
