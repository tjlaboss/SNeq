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
	kguess:         float; initial guess for the eigenvalue
					[Default: 1.0]
	"""
	def __init__(self, quad, mesh, bcs, kguess=1.0):
		self.quad = quad
		self.mesh = mesh
		self.k = kguess
		self.fission_source = np.ones(self.mesh.nx)
		self._get_psi_left, self._get_psi_right = self.__set_bcs(bcs)
	
	
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
			#raise NotImplementedError("periodic boundary condition")
			get_left = lambda n, g: self.mesh.psi[-1, n, g]
			get_right = lambda n, g: self.mesh.psi[0, n, g]
			return get_left, get_right
		
		if lc == "reflective":
			def get_left(n, g):
				m = self.quad.reflect_angle(n)
				return self.mesh.psi[0, m, g]
		elif lc == "vacuum":
			# No flux incoming from left
			get_left = lambda n, g: 0
		else:
			raise NotImplementedError(lc)
		
		if rc == "reflective":
			def get_right(n, g):
				m = self.quad.reflect_angle(n)
				return self.mesh.psi[-1, m, g]
		elif rc == "vacuum":
			# No flux incoming from right
			get_right = lambda n, g: 0
		else:
			raise NotImplementedError(rc)
		
		return get_left, get_right
	
	
	def transport_sweep(self, k):
		"""Perform one forward and one backward transport sweep.
		
		Returns:
		--------
		float; the L2 engineering norm after this sweep
		"""
		old_flux = np.array(self.mesh.flux[:, :])
		# TODO: Make the multiple energy groups do anything
		for g in range(self.mesh.groups):
			# Forward sweep
			for n in range(self.quad.N2):
				psi_in = self._get_psi_left(n, g)
				self.mesh.psi[0, n] = psi_in
				for i in range(self.mesh.nx):
					node = self.mesh.nodes[i]
					psi_out = node.flux_out(psi_in, n, g, k)
					self.mesh.psi[i+1, n] = psi_out
					psi_in = psi_out
			
			
			# Backward sweep
			for n in range(self.quad.N2, self.quad.N):
				psi_in = self._get_psi_right(n, g)
				self.mesh.psi[-1, n] = psi_in
				for i in range(self.mesh.nx):
					node = self.mesh.nodes[-1-i]
					psi_out = node.flux_out(psi_in, n, g, k)
					self.mesh.psi[-2-i, n] = psi_out
					psi_in = psi_out
				
			# Update the scalar flux using the Diamond Difference approximation
			#
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
				flux_left += w*(self.mesh.psi[0, n, g] + self._get_psi_left(n, g))/2.0
				flux_right += w*(self.mesh.psi[-1, n, g] + self._get_psi_right(n, g))/2.0
			self.mesh.flux[0, g] = flux_left
			self.mesh.flux[-1, g] = flux_right
			
		self.mesh.update_nodal_fluxes()
		
		# Get the fission source and flux differences
		fluxdiff = 0.0
		fsdiff = 0.0
		s_new = np.zeros(self.mesh.nx)
		for g in range(self.mesh.groups):
			for i in range(self.mesh.nx):
				phi_i1 = self.mesh.flux[i, g]
				phi_i0 = old_flux[i, g]
				if phi_i1 != phi_i0:
					fluxdiff += ((phi_i1 - phi_i0)/phi_i1)**2
				# And get the new fission source
				fs1 = self.mesh.nodes[i].get_fission_source(g, 1)
				fs0 = self.fission_source[i]
				s_new[i] += fs1
				fsdiff += ((fs1 - fs0)/fs1)**2
		rms_flux = np.sqrt(fluxdiff/self.mesh.nx)
		rms_fs = np.sqrt(fsdiff/self.mesh.nx)
		
		
		
		return s_new, rms_fs, rms_flux
			
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
		fsdiff = eps + 1
		fluxdiff = eps + 1
		count = 0
		# Outer: converge the fission source
		while fsdiff > eps:
			
			while fluxdiff > eps:
				fs, fsdiff, fluxdiff = self.transport_sweep(self.k)
				# Inner: converge the flux
				# Find the relative difference in flux using the L2 engineering norm
				
				count += 1
				if count >= maxiter:
					errstr = "Solution did NOT converge after {} iterations; aborting."
					warn(errstr.format(count))
					return self.mesh.flux
			
			# Now that flux has been converged, guess a new k
			# and update the fission source
			# Also find the relative difference in k
			print(self.fission_source, "->", fs)
			k_new = fs.sum()/self.fission_source.sum()
			kdiff = abs(k_new - self.k)/k_new
			self.fission_source = fs
			print("k: {}\tkdiff: {}".format(k_new, kdiff))
			
		
		print("Solution converged after {} iterations.".format(count))
		return self.mesh.flux
		
