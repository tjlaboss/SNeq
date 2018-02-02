# Calculator
#
# Classes and methods to calculate fluxes

import numpy as np
from warnings import warn
from constants import BOUNDARY_CONDITIONS

class DiamondDifferenceCalculator(object):
	"""Constructor class for diamond difference solvers

	Parameters:
	-----------
	quad:           Quadrature; 1-D angular quadrature to use
	mesh:           Mesh1D; 1-D mesh to solve on.
					Use one tailored to your problem.
	bcs:            tuple of ("left", "right")
	kguess:         float; initial guess for the eigenvalue
					[Default: 1.0]
	accelerator:    Accelerator, if one is desired
					[Default: None]
	tallies:        Tally, if any are desired
					[Default: None]
	"""
	
	def __init__(self, quad, mesh, bcs, kguess=1.0, accelerator=None, tallies=None):
		self.quad = quad
		self.mesh = mesh
		self.k = kguess
		self.accelerator = accelerator
		if tallies is None:
			self.tallies = []
		else:
			self.tallies = tallies
		self.fission_source = self.mesh.calculate_fission_source()
		self.scatter_source = self.mesh.calculate_scatter_source()
	
	def _set_bcs(self, bcs):
		# TODO: CORNER CASES
		for bc in bcs:
			assert bc in BOUNDARY_CONDITIONS, \
				"{} is an unknown boundary condition.".format(bc)
		nd = len(bcs) // 2
		
		# East and west sides
		wc, ec = bcs[0:2]
		if "periodic" in (wc, ec) and wc != ec:
			errstr = "If one edge has a periodic boundary condition, " \
			         "both sides must."
			raise TypeError(errstr)
		
		if wc == "periodic":
			if nd == 1:
				get_west = lambda n, g: self.mesh.psi[-1, n, g]
			elif nd == 2:
				get_west = lambda j, n, g: self.mesh.psi_east[j, n, g]
			else:
				raise NotImplementedError("{}D".format(nd))
		elif wc == "reflective":
			if nd == 1:
				def get_west(n, g):
					m = self.quad.reflect_angle(n)
					return self.mesh.psi[0, m, g]
			elif nd == 2:
				def get_west(j, n, g):
					m = self.quad.inverse_reflect_angle(n, "west")
					return self.mesh.psi_west[j, m, g]
			else:
				raise NotImplementedError("{}D".format(nd))
		elif wc == "vacuum":
			# No flux incoming from left
			if nd == 1:
				get_west = lambda n, g: 0
			elif nd == 2:
				get_west = lambda j, n, g: 0
			else:
				raise NotImplementedError("{}D".format(nd))
		else:
			raise NotImplementedError(wc)
		
		if ec == "periodic":
			if nd == 1:
				get_east = lambda n, g: self.mesh.psi[0, n, g]
			elif nd == 2:
				get_east = lambda j, n, g: self.mesh.psi_west[j, n, g]
			else:
				raise NotImplementedError("{}D".format(nd))
		elif ec == "reflective":
			if nd == 1:
				def get_east(n, g):
					m = self.quad.reflect_angle(n)
					return self.mesh.psi[-1, m, g]
			elif nd == 2:
				def get_east(j, n, g):
					m = self.quad.inverse_reflect_angle(n, "east")
					return self.mesh.psi_east[j, m, g]
			else:
				raise NotImplementedError("{}D".format(nd))

		elif ec == "vacuum":
			# No flux incoming from right
			if nd == 1:
				get_east = lambda n, g: 0
			elif nd == 2:
				get_east = lambda j, n, g: 0
			else:
				get_east = lambda j, k, n, g: 0
		else:
			raise NotImplementedError(ec)
		
		if nd == 1:
			return get_west, get_east
			
		# North and south sides
		nc, sc = bcs[2:4]
		if "periodic" in (nc, sc) and nc != sc:
			errstr = "If one edge has a periodic boundary condition, " \
			         "both sides must."
			raise TypeError(errstr)
		if nc == "periodic":
			if nd == 2:
				get_north = lambda i, n, g: self.mesh.psi_south[i, n, g]
			else:
				raise NotImplementedError("{}D".format(nd))
		elif nc == "reflective":
			if nd == 2:
				def get_north(i, n, g):
					m = self.quad.inverse_reflect_angle(n, "north")
					return self.mesh.psi_north[i, m, g]
			else:
				raise NotImplementedError("{}D".format(nd))
		elif nc == "vacuum":
			# No flux incoming from top
			if nd == 2:
				get_north = lambda i, n, g: 0
			else:
				get_north = lambda i, k, n, g: 0
		else:
			raise NotImplementedError(nc)
		
		if sc == "periodic":
			if nd == 2:
				get_south = lambda i, n, g: self.mesh.psi_north[i, n, g]
			else:
				raise NotImplementedError("{}D".format(nd))
		elif sc == "reflective":
			if nd == 2:
				def get_south(i, n, g):
					m = self.quad.inverse_reflect_angle(n, "south")
					return self.mesh.psi_south[i, m, g]
			else:
				raise NotImplementedError("{}D".format(nd))
		elif sc == "vacuum":
			if nd == 2:
				get_south = lambda i, n, g: 0
			else:
				get_south = lambda i, k, n, g: 0
		else:
			raise NotImplementedError(sc)
		
		if nd == 2:
			return get_west, get_east, get_north, get_south

	def transport_sweep(self, k):
		pass


class DiamondDifferenceCalculator1D(DiamondDifferenceCalculator):
	"""Multigroup, one-dimensional diamond difference solver
	
	Parameters:
	-----------
	quad:           Quadrature; 1-D angular quadrature to use
	mesh:           Mesh1D; 1-D mesh to solve on.
					Use one tailored to your problem.
	bcs:            tuple of ("left", "right")
	kguess:         float; initial guess for the eigenvalue
					[Default: 1.0]
	accelerator:    Accelerator, if one is desired
					[Default: None]
	tallies:        Tally1D, if any are desired
					[Default: None]
	"""
	def __init__(self, quad, mesh, bcs, kguess=1.0, accelerator=None, tallies=None):
		assert len(bcs) == 2, "A 1D solution requires 2 boundary conditions."
		super().__init__(quad, mesh, bcs, kguess, accelerator, tallies)
		self._get_psi_west, self._get_psi_east = self._set_bcs(bcs)
	
	
	def _get_source(self, i, g, k=None):
		source = self.mesh.nodes[i].source
		source += 0.5*self.scatter_source[i, g]
		if k:
			source += 0.5*self.fission_source[i, g]/k
		return source
	
	def transport_sweep(self, k):
		"""Perform one forward and one backward transport sweep.
		
		Returns:
		--------
		float; the L2 engineering norm after this sweep
		"""
		old_flux = np.array(self.mesh.flux[:, :])
		for g in range(self.mesh.groups):
			# Forward sweep
			for n in range(self.quad.N2):
				mu = abs(self.quad.mus[n])
				psi_in = self._get_psi_west(n, g)
				self.mesh.psi[0, n] = psi_in
				for i in range(self.mesh.nx):
					node = self.mesh.nodes[i]
					q = self._get_source(i, g, k)
					psi_out = psi_in*(2*mu - node.dx*node.sigma_tr[g]) + 2*node.dx*q
					psi_out /= 2*mu + node.dx*node.sigma_tr[g]
					
					self.mesh.psi[i+1, n] = psi_out
					psi_in = psi_out
			
			
			# Backward sweep
			for n in range(self.quad.N2, self.quad.N):
				mu = abs(self.quad.mus[n])
				psi_in = self._get_psi_east(n, g)
				self.mesh.psi[-1, n, g] = psi_in
				for i in range(self.mesh.nx):
					node = self.mesh.nodes[-1-i]
					q = self._get_source(i, g, k)
					psi_out = psi_in*(2*mu - node.dx*node.sigma_tr[g]) + 2*node.dx*q
					psi_out /= 2*mu + node.dx*node.sigma_tr[g]
					
					self.mesh.psi[-2-i, n, g] = psi_out
					psi_in = psi_out
				
				# Reconnect that last pesky boundary flux
				self.mesh.psi[0, n, g] = psi_out
				
			# Update the scalar flux using the Diamond Difference approximation
			#
			# Interior nodes
			for i in range(self.mesh.nx):
				flux_i = 0.0
				for n in range(self.quad.N):
					w = self.quad.weights[n]
					psi_plus = self.mesh.psi[i+1, n, g]
					psi_minus = self.mesh.psi[i, n, g]
					flux_i += w*(psi_plus + psi_minus)/2.0
				self.mesh.flux[i, g] = flux_i
		
		
		# Get the fission source and flux differences
		fluxdiff = 0.0
		
		if self.fission_source is not None:
			fsdiff = 0.0
			fs_new = self.mesh.calculate_fission_source()
			for i in range(self.mesh.nx):
				for g in range(self.mesh.groups):
					# Calculate the fission source difference
					fs0 = self.fission_source[i, g]
					fs1 = fs_new[i, g]
					if fs1 != fs0:
						fsdiff += ((fs1 - fs0)/fs1)**2
			rms_fs = np.sqrt(fsdiff/self.mesh.nx)
		else:
			rms_fs = 0.0
			fs_new = 0
		
		
		for i in range(self.mesh.nx):
			for g in range(self.mesh.groups):
				# Calculate the flux difference
				phi_i1 = self.mesh.flux[i, g]
				phi_i0 = old_flux[i, g]
				if phi_i1 != phi_i0:
					fluxdiff += ((phi_i1 - phi_i0)/phi_i1)**2
		rms_flux = np.sqrt(fluxdiff/self.mesh.nx)
		
		
		return fs_new, rms_fs, rms_flux
		
				
	def solve(self, eps, test_convergence=lambda: True, maxiter=1000):
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
		kdiff = eps + 1
		outer_count = 0
		# Outer: converge the fission source
		while (fsdiff > eps or kdiff > eps) or not test_convergence():
			print("kguess = {}".format(self.k))
			inner_count = 0
			fluxdiff = eps + 1
			
			if self.fission_source is not None:
				fsdiff = 0
				kdiff = 0
			
			while fluxdiff > eps:
				fs, fsdiff, fluxdiff = self.transport_sweep(self.k)
				# Inner: converge the flux
				# Find the relative difference in flux using the L2 engineering norm
				
				inner_count += 1
				if inner_count >= maxiter:
					errstr = "Solution did NOT converge after {} inner iterations; aborting."
					warn(errstr.format(inner_count))
					return False
				
				# TODO: Figure out if this is the right place to put the acceleration method
				if self.accelerator:
					# Update the accleration method with the fine mesh fluxes
					self.accelerator.restrict(self.mesh)
					# Converge the acceleration flux
					print("Coarse mesh iterations would go here.")
					# Update our fine mesh solution from the coarse mesh
					self.accelerator.prolong(self.mesh)
				
				#print("Inner Iter {}: flux, rms = {}".format(inner_count, fluxdiff))
				#print(self.mesh.flux)
			
			print("Outer Iteration {}: flux converged at kguess = {}".format(outer_count, self.k))
			print(self.mesh.flux)
			
			
			# Now that flux has been converged, guess a new k
			# and update the fission source
			# Also find the relative difference in k
			ss = self.mesh.calculate_scatter_source()
			if self.fission_source is not None:
				print(self.fission_source, "->", fs)
				k_new = self.k*fs.sum()/self.fission_source.sum()
				kdiff = abs(k_new - self.k)/k_new
				print("k: {}\tkdiff: {}".format(k_new, kdiff))
				self.fission_source = fs
				self.k = k_new
				print("\n\n")
			self.scatter_source = ss
			
			
			outer_count += 1
			if outer_count >= maxiter:
				errstr = "Solution did NOT converge after {} outer iterations; aborting."
				warn(errstr.format(outer_count))
				return False
			
		
		print("Solution converged after {} outer iterations.".format(outer_count))
		return True
		

class DiamondDifferenceCalculator2D(DiamondDifferenceCalculator):
	"""One-group, two-dimensional diamond difference solver
	
	This class currently saves the angular fluxes. It doesn't have to,
	but it makes development and debugging far easier.
	
	Parameters:
	-----------
	quad:           Quadrature; 2-D angular quadrature to use
	mesh:           Mesh2D; 2-D mesh to solve on.
					Use one tailored to your problem.
	bcs:            tuple of ("west", "east", "north", "south")
	kguess:         float; initial guess for the eigenvalue
					[Default: 1.0]
	tallies:        Tally2D, if any are desired
					[Default: None]
	"""
	def __init__(self, quad, mesh, bcs, kguess=1.0, accelerator=None, tallies=None):
		super().__init__(quad, mesh, bcs, kguess, accelerator, tallies)
		self._get_psi_west, self._get_psi_east, self._get_psi_north, \
			self._get_psi_south = self._set_bcs(bcs)
		
	
	def _get_source(self, i, j, g, k=None):
		"""Get the fixed and scattering source at a given location.
		Fission source is ignored, but will be enabled
		for the eigenvalue problem when that is implemented.
		
		Parameters:
		----------
		i:          int; x-index
		j:          int; y-index
		g:          int; energy group index
		k:          float; estimate for the eigenvalue
					[Not implemented]
		
		Returns:
		--------
		source:     float; total source at (i, j)
		"""
		source = self.mesh.nodes[i, j].source
		source += 0.5*self.scatter_source[i, j, g]
		#if k:
		#	source += 0.5*self.fission_source[i, j, g]/k
		return source
	
	def l2norm2d(self, new_array, old_array):
		rms = np.sqrt(( ((new_array - old_array)/new_array)**2 ).sum()/self.mesh.nx/self.mesh.ny)
		return rms
	
	def transport_sweep(self, k=None):
		"""Perform a round of 2D transport sweeps.
		
		1. Forward (from the top left)
		2. Forward (from the bottom left)
		3. Backward (from the top right)
		4. Backward (from the bottom right)
		
		Parameter:
		----------
		k:          float; guess for the eigenvalue.
					Currently not hooked up to anything; it's just a placebo.
					[Default: None]
		
		Returns:
		--------
		float; RMS of the flux after the sweep
		"""
		old_flux = np.array(self.mesh.flux[:, :, :])
		self.mesh.flux = np.zeros(old_flux.shape)
		for g in range(self.mesh.groups):
			# Forward sweep from top left:
			# +mux, -muy, 0 -> npq
			for n in range(self.quad.npq):
				mux = abs(self.quad.muxs[n])
				muy = abs(self.quad.muys[n])
				
				# FIXME: Not sure which level index is correct.
				# p = n // 4
				p = n % self.quad.npq
				w = self.quad.weights[p]
				
				psi_w = np.array([self._get_psi_west(j, n, g) for j in range(self.mesh.ny)])
				for i in range(self.mesh.nx):
					psi_in_n = self._get_psi_north(i, n, g)
					self.mesh.psi_north[i, n, g] = psi_in_n
					for j in range(self.mesh.ny):
						psi_in_w = psi_w[j]
						node = self.mesh.nodes[i, j]
						q = self._get_source(i, j, g, k)
						xcoeff = 2*mux/node.dx
						ycoeff = 2*muy/node.dy
						psi_bar = (q + xcoeff*psi_in_w + ycoeff*psi_in_n) / \
						          (node.sigma_tr[g] + xcoeff + ycoeff)
						psi_out_e = 2*psi_bar - psi_in_w
						psi_out_s = 2*psi_bar - psi_in_n
						psi_w[j] = psi_out_e
						if i == self.mesh.nx - 1:
							self.mesh.psi_east[j, n, g] = psi_out_e
						psi_in_n = psi_out_s
						# Tally the results
						for tal in self.tallies:
							if tal.applies(i, j, n, g):
								tal.update(psi_bar, n, g)
						# Update the scalar flux
						self.mesh.flux[i, j, g] += w*(psi_in_w + psi_out_e)/2.0
					self.mesh.psi_south[i, n, g] = psi_out_s
						
			# Forward sweep from bottom left:
			# +mux, +muy, npq -> 2*npq
			for n in range(self.quad.npq, 2*self.quad.npq):
				mux = abs(self.quad.muxs[n - self.quad.npq])
				muy = abs(self.quad.muys[n - self.quad.npq])
				p = n%self.quad.npq
				w = self.quad.weights[p]
				psi_w = np.array([self._get_psi_west(j, n, g) for j in range(self.mesh.ny)])
				for i in range(self.mesh.nx):
					psi_in_s = self._get_psi_south(i, n, g)
					self.mesh.psi_south[i, n, g] = psi_in_s
					for j in reversed(range(self.mesh.ny)):
						psi_in_w = psi_w[j]
						node = self.mesh.nodes[i, j]
						q = self._get_source(i, j, g, k)
						xcoeff = 2*mux/node.dx
						ycoeff = 2*muy/node.dy
						psi_bar = (q + xcoeff*psi_in_w + ycoeff*psi_in_s)/ \
						          (node.sigma_tr[g] + xcoeff + ycoeff)
						psi_out_e = 2*psi_bar - psi_in_w
						psi_out_n = 2*psi_bar - psi_in_s
						psi_w[j] = psi_out_e
						if i == self.mesh.nx - 1:
							self.mesh.psi_east[j, n, g] = psi_out_e
						psi_in_s = psi_out_n
						# Tally the results
						for tal in self.tallies:
							if tal.applies(i, j, n, g):
								tal.update(psi_bar, n, g)
						# Update the scalar flux
						self.mesh.flux[i, j, g] += w*(psi_in_w + psi_out_e)/2.0
					self.mesh.psi_north[i, n, g] = psi_out_n
			
			# Backward sweep from bottom right:
			# -mux, +muy, 2*npq -> 3*npq
			for n in range(2*self.quad.npq, 3*self.quad.npq):
				mux = abs(self.quad.muxs[n - 2*self.quad.npq])
				muy = abs(self.quad.muys[n - 2*self.quad.npq])
				p = n%self.quad.npq
				w = self.quad.weights[p]
				psi_e = np.array([self._get_psi_east(j, n, g) for j in range(self.mesh.ny)])
				for i in reversed(range(self.mesh.nx)):
					psi_in_s = self._get_psi_south(i, n, g)
					self.mesh.psi_south[i, n, g] = psi_in_s
					for j in reversed(range(self.mesh.ny)):
						psi_in_e = psi_e[j]
						node = self.mesh.nodes[i, j]
						q = self._get_source(i, j, g, k)
						xcoeff = 2*mux/node.dx
						ycoeff = 2*muy/node.dy
						psi_bar = (q + xcoeff*psi_in_e + ycoeff*psi_in_s)/ \
						          (node.sigma_tr[g] + xcoeff + ycoeff)
						psi_out_w = 2*psi_bar - psi_in_e
						psi_out_n = 2*psi_bar - psi_in_s
						psi_e[j] = psi_out_w
						if i == 0:
							self.mesh.psi_west[j, n, g] = psi_out_w
						psi_in_s = psi_out_n
						# Tally the results
						for tal in self.tallies:
							if tal.applies(i, j, n, g):
								tal.update(psi_bar, n, g)
						# Update the scalar flux
						self.mesh.flux[i, j, g] += w*(psi_in_e + psi_out_w)/2.0
					self.mesh.psi_north[i, n, g] = psi_out_n
			
			# Backward sweep from the top right
			# -mux, -muy, 3*npq -> 4*npq
			for n in range(3*self.quad.npq, self.quad.Nflux):
				mux = abs(self.quad.muxs[n - 3*self.quad.npq])
				muy = abs(self.quad.muys[n - 3*self.quad.npq])
				p = n % self.quad.npq
				w = self.quad.weights[p]
				psi_e = np.array([self._get_psi_east(j, n, g) for j in range(self.mesh.ny)])
				for i in reversed(range(self.mesh.nx)):
					psi_in_n = self._get_psi_north(i, n, g)
					self.mesh.psi_north[i, n, g] = psi_in_n
					for j in range(self.mesh.ny):
						psi_in_e = psi_e[j]
						node = self.mesh.nodes[i, j]
						q = self._get_source(i, j, g, k)
						xcoeff = 2*mux/node.dx
						ycoeff = 2*muy/node.dy
						psi_bar = (q + xcoeff*psi_in_e + ycoeff*psi_in_n)/ \
						          (node.sigma_tr[g] + xcoeff + ycoeff)
						psi_out_w = 2*psi_bar - psi_in_e
						psi_out_s = 2*psi_bar - psi_in_n
						psi_e[j] = psi_out_w
						if i == 0:
							self.mesh.psi_west[j, n, g] = psi_out_w
						psi_in_n = psi_out_s
						# Tally the results
						for tal in self.tallies:
							if tal.applies(i, j, n, g):
								tal.update(psi_bar, n, g)
						# Update the scalar flux
						self.mesh.flux[i, j, g] += w*(psi_in_e + psi_out_w)/2.0
					self.mesh.psi_south[i, n, g] = psi_out_s
			
		rms_flux = self.l2norm2d(self.mesh.flux, old_flux)
		return rms_flux
	

	def solve(self, eps, test_convergence = lambda: True, maxiter=1000):
		outer_count = 0
		ssdiff = 1 + eps
		while (ssdiff > eps) or not test_convergence():
			old_source = np.array(self.scatter_source)
			fluxdiff = self.transport_sweep(self.k)
			'''
			while fluxdiff > eps:
				fluxdiff = self.transport_sweep(self.k)
				# Inner: converge the flux
				
				inner_count += 1
				if inner_count >= maxiter:
					errstr = "Solution did NOT converge after {} inner iterations; aborting."
					warn(errstr.format(inner_count))
					return False
				
				print("rms:", fluxdiff)
			
			print("Converged after {} inner iterations.".format(inner_count))
			'''
			
			outer_count += 1
			if outer_count >= maxiter:
				errstr = "Solution did NOT converge after {} outer iterations; aborting."
				warn(errstr.format(outer_count))
				return False
			
			self.mesh.flux /= self.mesh.flux.mean()
			ss = self.mesh.calculate_scatter_source()
			ssdiff = self.l2norm2d(ss, old_source)
			print("RMS (outer): {}".format(ssdiff))
			self.scatter_source = ss
			
		print("Solution converged after {} outer iterations".format(outer_count))
		return True