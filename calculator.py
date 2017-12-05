# Calculator
#
# Classes and methods to calculate fluxes

import numpy as np
from warnings import warn
from constants import BOUNDARY_CONDITIONS

class DiamondDifferenceCalculator1D(object):
	"""One-group, one-dimensional diamond difference solver
	
	Parameters:
	-----------
	quad:           Quadrature; 1-D angular quadrature to use
	mesh:           Mesh1D; 1-D mesh to solve on.
					Use one tailored to your problem.
	bcs:            tuple of ("left", "right")
	"""
	def __init__(self, quad, mesh, bcs):
		self.quad = quad
		self.mesh = mesh
		self._psi_left, self._psi_right = self.__set_bcs(bcs)
	
	
	def __set_bcs(self, bcs):
		for bc in bcs:
			assert bc in BOUNDARY_CONDITIONS, \
				"{} is an unknown boundary condition.".format(bc)
		lc, rc = bcs
		if "periodic" in bcs and lc != rc:
			errstr = "If one edge has a periodic boundary condition, " \
			         "both sides must."
			raise TypeError(errstr)
		
		if "periodic" in bcs:
			raise NotImplementedError("periodic boundary condition")
			left = lambda n, g: self.mesh.psi[-1, n, g]
			right = lambda n, g: self.mesh.psi[0, n, g]
			return left, right
		
		if lc == "reflective":
			# FIXME: This doesn't work
			left = lambda n, g: self.mesh.psi[1, n, g]
		elif lc == "vacuum":
			# No flux incoming from left
			left = lambda n, g: 0
		else:
			raise NotImplementedError(lc)
		
		if rc == "reflective":
			# FIXME: This doesn't work
			right = lambda n, g: self.mesh.psi[-2, n, g]
		elif rc == "vacuum":
			# No flux incoming from right
			right = lambda n, g: 0
		else:
			raise NotImplementedError(rc)
		
		return left, right
	
	
	def transport_sweep(self):
		"""Perform one forward and one backward transport sweep.
		
		Returns:
		--------
		float; the L2 engineering norm after this sweep
		"""
		old_flux = np.array(self.mesh.flux[:, :])
		N2 = self.quad.N//2
		
		# TODO: Make the multiple energy groups do anything
		for g in range(self.mesh.groups):
			# Forward sweep
			for n in range(N2):
				psi_in = self._psi_left(n, g)
				for i in range(self.mesh.nx):
					node = self.mesh.nodes[i]
					psi_out = node.flux_out(psi_in, n, g)
					self.mesh.psi[i, n] = psi_out
			
			# Backward sweep
			for n in range(N2, self.quad.N):
				psi_in = self._psi_right(n, g)
				for i in range(self.mesh.nx):
					node = self.mesh.nodes[-1-i]
					psi_out = node.flux_out(psi_in, n, g)
					self.mesh.psi[-1-i, n] = psi_out
			
			# Update the scalar flux using the Diamond Difference approximation
			
			# Interior nodes
			for i in range(1, self.mesh.nx - 1):
				flux_i = 0.0
				for n in range(self.quad.N):
					w = self.quad.weights[n]
					psi_plus = self.mesh.psi[i+1, n]
					psi_minus = self.mesh.psi[i, n]
					flux_i += w*(psi_plus + psi_minus)/2.0
				self.mesh.flux[i, g] = flux_i
		
			# Boundary nodes
			flux_left = 0.0
			flux_right = 0.0
			for n in range(self.quad.N):
				w = self.quad.weights[n]
				flux_left += w*(self.mesh.psi[0, n, g] + self._psi_left(n, g))/2.0
				flux_right += w*(self.mesh.psi[-1, n, g] + self._psi_right(n, g))/2.0
			self.mesh.flux[0, g] = flux_left
			self.mesh.flux[-1, g] = flux_right
			
		self.mesh.update_nodal_fluxes()
			
		# Find the relative difference using the L2 engineering norm
		diff = 0.0
		for g in range(self.mesh.groups):
			for i in range(self.mesh.nx):
				phi_i1 = self.mesh.flux[i, g]
				phi_i0 = old_flux[i, g]
				if phi_i1 != phi_i0:
					diff += ((phi_i1 - phi_i0)/phi_i1)**2
		
		return np.sqrt(diff/self.mesh.nx)
			
	def solve(self, eps, maxiter=1000):
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
			#print(count, ":\t", diff)
			
			count += 1
			if count >= maxiter:
				errstr = "Solution did NOT converge after {} iterations; aborting."
				warn(errstr.format(count))
				return self.mesh.flux
		
		print("Solution converged after {} iterations.".format(count))
		return self.mesh.flux
		
