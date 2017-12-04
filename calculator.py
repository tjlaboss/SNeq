# Calculator
#
# Classes and methods to calculate fluxes

import numpy as np
from warnings import warn

class DiamondDifferenceCalculator1D(object):
	"""One-group, one-dimensional diamond difference solver
	
	Parameters:
	-----------
	quad:           Quadrature; 1-D angular quadrature to use
	mesh:           Mesh1D; 1-D mesh to solve on.
					Use one tailored to your problem.
	"""
	def __init__(self, quad, mesh):
		self.quad = quad
		self.mesh = mesh
	
	
	def transport_sweep(self):
		"""Perform one forward and one backward transport sweep.
		
		Returns:
		--------
		float; the L2 engineering norm after this sweep
		"""
		# TODO: Align the iterations and boundary conditions
		old_flux = self.mesh.flux[:]
		N2 = self.quad.N//2
		
		# Forward sweep
		for n in range(N2):
			psi_in = 0.0
			for i in range(self.mesh.nx):
				node = self.mesh.nodes[i]
				psi_out = node.flux_out(psi_in, n)
				self.mesh.psi[i, n] = psi_out
		
		# Backward sweep
		for n in range(N2+1, self.quad.N):
			psi_in = 0.0
			for i in range(self.mesh.nx):
				node = self.mesh.nodes[-1-i]
				psi_out = node.flux_out(psi_in, n)
				self.mesh.psi[-1-i, n] = psi_out
		
		# Update the scalar flux using the Diamond Difference approximation
		for i in range(self.mesh.nx):
			flux_i = 0.0
			for n in range(self.quad.N):
				w = self.quad.weights[n]
				psi_plus = self.mesh.psi[i+1, n]
				psi_minus = self.mesh.psi[i, n]
				flux_i += w*(psi_plus + psi_minus)/2.0
			self.mesh.flux[i] = flux_i
		self.mesh.update_nodal_fluxes()
		
		# Find the relative difference using the L2 engineering norm
		diff = 0.0
		for i in range(self.mesh.nx):
			diff += (self.mesh.flux[i] - old_flux[i])**2
		
		return np.sqrt(diff/self.mesh.nx)
			
	def solve(self, eps, maxiter=100):
		"""Solve on the mesh within tolerance
		
		Parameters:
		-----------
		eps:            float; tolerance to use
		maxiter:        int; the maximum number of iterations
						[Default: 100]
		
		Returns:
		--------
		flux:           numpy array of the scalar flux
		"""
		diff = eps + 1
		count = 0
		while diff > eps:
			diff = self.transport_sweep()
			
			count += 1
			if count >= maxiter:
				errstr = "Solution did NOT converge after {} iterations; aborting."
				warn(errstr.format(count))
				return self.mesh.flux
		
		print("Solution converged after {} iterations.".format(count))
		return self.mesh.flux
		
