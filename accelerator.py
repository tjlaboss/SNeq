# Accelerator
#
# There are various acceleration methods one can use with discrete ordinates
# radiation transport. The Accelerator and its derived classes serve as
# wrappers for acceleration methods of various colors and shapes.

import diffusion
from constants import BOUNDARY_CONDITIONS
import scipy
import scipy.linalg as linalg

class Accelerator(object):
	"""Base class for acceleration wrappers
	
	
	"""
	def __init__(self, coarse_mesh, fine_mesh):
		self.coarse_mesh = coarse_mesh
		self.fine_mesh = fine_mesh
		if coarse_mesh.bcs is not None:
			assert coarse_mesh.bcs == fine_mesh.bcs, \
				"Coarse mesh and fine mesh boundary conditions do not match."
		for bc in fine_mesh.bcs:
			assert bc in BOUNDARY_CONDITIONS, \
				"{} is an unknown boundary condition.".format(bc)
		self.bcs = fine_mesh.bcs
	
	def restrict_flux(self, fine_mesh):
		"""Integrate the fine mesh onto the coarse mesh."""
		self.coarse_mesh.restrict_flux(fine_mesh)
	
	def restrict_source(self, fine_ss, fine_fs, k):
		self.coarse_mesh.get_coarse_source(fine_ss, fine_ss, fine_fs, k)
	
	def prolong(self):
		"""Update the fine mesh using the coarse mesh."""
		self.coarse_mesh.prolong_flux(self.fine_mesh, self.coarse_mesh.flux)
	
	def solve(self, ss, fs, k, eps):
		"""Get the coarse mesh solution
		
		Parameter:
		----------
		eps:        float; relative tolerance
		"""
		pass


class RebalanceAccelerator1D(Accelerator):
	"""An Accelerator for the 1-D Coarse Mesh Rebalance method"""
	def __init__(self, coarse_mesh, fine_mesh):
		super().__init__(coarse_mesh, fine_mesh)
		self.rebalance_factors = scipy.ones(self.coarse_mesh.nx*self.coarse_mesh.groups)
		
	
	def solve(self, ss, fs, k, eps):
		"""Solve the CMR problem for the rebalance factors"""
		G = self.coarse_mesh.groups
		NX = self.coarse_mesh.nx
		NC = NX*G
		RAT = self.coarse_mesh.ratio
		matA = scipy.zeros((NC, NC))
		
		# Set up the destruction matrix, [A]
		for g in range(G):
			i0 = g*NX
			# Interior nodes
			for cmi in range(1, NX-1):
				l = -self.coarse_mesh.currents[0, cmi, g]
				r = -self.coarse_mesh.currents[1, cmi+1, g]
				'''
				# TODO: Determine whether to just use the coarse mesh flux here
				transport = 0
				for fmi in range(RAT):
					i = cmi*RAT + fmi
					node = self.fine_mesh.nodes[i]
					phi = self.fine_mesh.flux[i, g]
					transport += node.sigma_tr[g]*phi#*node.dx
				'''
				node = self.coarse_mesh.nodes[cmi]
				phidx = self.coarse_mesh.flux[cmi, g]
				transport = node.sigma_tr[g]*phidx#*node.dx
				
				c = +self.coarse_mesh.currents[0, cmi+1, g] + \
				     self.coarse_mesh.currents[1, cmi, g] + transport
				matA[i0+cmi, (i0+cmi-1):(i0+cmi+2)] = [l, c, r]
			# Boundary nodes
			bcl, bcr = self.bcs
			# Leftmost node
			r = -self.coarse_mesh.currents[1, 1, g]
			'''
			transport = 0
			for fmi in range(RAT):
				i = fmi
				node = self.fine_mesh.nodes[i]
				phi = self.fine_mesh.flux[i, g]
				transport += node.sigma_tr[g]*phi#*node.dx
			'''
			node = self.coarse_mesh.nodes[0]
			phidx = self.coarse_mesh.flux[0, g]
			transport = node.sigma_tr[g]*phidx
			if bcl == "reflective":
				c = self.coarse_mesh.currents[0, 0, g] + transport
				# or should this be  currents[0, 1, g]??
				# I think it's handled by the boundary condition.
			elif bcl == "vacuum":
				c = +self.coarse_mesh.currents[0, 1, g] + \
				     self.coarse_mesh.currents[1, 0, g] + transport
			else:
				raise NotImplementedError(bcl)
			matA[i0, i0:(i0 + 2)] = [c, r]
			# Rightmost node
			l = -self.coarse_mesh.currents[0, -2, g]
			node = self.coarse_mesh.nodes[-1]
			phidx = self.coarse_mesh.flux[-1, g]
			transport = node.sigma_tr[g]*phidx
			if bcr == "reflective":
				c = self.coarse_mesh.currents[1, -1, g] + transport
			elif bcr == "vacuum":
				c = +self.coarse_mesh.currents[0, -1, g] + \
					 self.coarse_mesh.currents[1, -2, g] + transport
			else:
				raise NotImplementedError(bcr)
			matA[i0+NX-1, (i0+NX-2):(i0+NX)] = [l, c]
		#print(matA)
		
		# Set up the creation matrix, [B]
		matB = self.coarse_mesh.get_coarse_source(self.fine_mesh, ss, fs, k)
		#matB /= matB.sum() # debug: normalize?
		# CMR
		self.rebalance_factors = linalg.solve(matA, matB)
		# This is one long vector; reshape it into a multigroup array.
		self.rebalance_factors.shape = (self.coarse_mesh.groups, self.coarse_mesh.nx)
		self.rebalance_factors = self.rebalance_factors.T
		print("Rebalance factors:")
		#self.rebalance_factors /= self.rebalance_factors.mean()
		print(self.rebalance_factors)
		print()
	
			
	def prolong(self):
		"""Use the coarse mesh flux to update the fine mesh flux.
		This RebalancePincell1D's flux will be updated to the
		coarse flux after the prolongation is complete.

		Parameter:
		----------
		fine_mesh:      Pincell1D instance
		coarse_flux:    array of the new coarse flux to use for the prolongation.
		"""
		return self.coarse_mesh.prolong_flux(self.fine_mesh, self.rebalance_factors)
		


class FiniteDifference1D(Accelerator):
	"""An Accelerator for the 1-D CMFD method"""
	def __init__(self, coarse_mesh, fine_mesh):
		super().__init__(coarse_mesh, fine_mesh)
		
		# Define the first set of coupling coefficients on the mesh
		self.dtildes = scipy.zeros((2, self.coarse_mesh.nx, self.coarse_mesh.groups))
		for g in range(self.coarse_mesh.groups):
			# Interior nodes
			for i in range(1, self.coarse_mesh.nx - 1):
				nl = self.coarse_mesh.nodes[i - 1]
				nc = self.coarse_mesh.nodes[i]
				nr = self.coarse_mesh.nodes[i + 1]
				self.dtildes[0, i, g] = 2*nc.D[g]*nr.D[g]/(nc.D[g]*nr.dx + nr.D[g]*nc.dx)
				self.dtildes[1, i, g] = 2*nc.D[g]*nl.D[g]/(nc.D[g]*nl.dx + nl.D[g]*nc.dx)
			# Boundary nodes
			bcl, bcr = self.bcs
			# Left edge
			nc = self.coarse_mesh.nodes[0]
			if bcl == "reflective":
				self.dtildes[1, 0, g] = 0
			elif bcl == "vacuum":
				al = 0.5
				self.dtildes[1, 0, g] = -2*al*nc.D[g]/(al*nc.dx + 2*nc.D[g])
			else:
				raise NotImplementedError(bcl)
			# Right edge
			nc = self.coarse_mesh.nodes[-1]
			if bcr == "reflective":
				self.dtildes[0, -1, g] = 0
			elif bcr == "vacuum":
				ar = 0.5
				self.dtildes[0, -1, g] = -2*ar*nc.D[g]/(ar*nc.dx + 2*nc.D[g])
			else:
				raise NotImplementedError(bcr)
			
			
		#print(self.dtildes)
	
	
	def _build_matrix_a(self, flux):
		"""Set up the destruction matrix, [A]"""
		G = self.coarse_mesh.groups
		NX = self.coarse_mesh.nx
		NC = NX*G
		matA = scipy.zeros((NC, NC))
		for g in range(G):
			i0 = g*NX
			# Interior nodes
			for cmi in range(1, NX - 1):
				node = self.coarse_mesh.nodes[cmi]
				cmidx = node.dx
				# Fluxes
				phil = flux[cmi-1, g]
				phic = flux[cmi, g]
				phir = flux[cmi+1, g]
				# Currents
				jplus =  -self.coarse_mesh.currents[0, cmi, g]
				jminus = -self.coarse_mesh.currents[1, cmi+1, g]
				# Coupling coefficients
				dsquigl = self.dtildes[0, cmi, g]
				dsquigr = self.dtildes[1, cmi, g]
				dhatl = -(jminus + dsquigl*(phil - phic))/(phil + phic)
				dhatr =  -(jplus + dsquigr*(phir - phic))/(phir + phic)
				
				# Matrix entries
				l = +(dhatl - dsquigr)/cmidx
				c = node.sigma_tr[g] + (dhatl - dhatr + dsquigl + dsquigr)/cmidx
				r = -(dhatr + dsquigr)/cmidx
				matA[i0 + cmi, (i0+cmi-1):(i0+cmi+2)] = [l, c, r]
			# Boundary nodes
			bcl, bcr = self.bcs
			# Leftmost
			node = self.coarse_mesh.nodes[0]
			cmidx = node.dx
			phic = flux[0, g]
			phir = flux[1, g]
			if bcl == "reflective":
				dsquigr = self.dtildes[1, 0, g]  # [1, 1, g]?
				# TODO: determine whether this is J+ or J-
				jplus =  -self.coarse_mesh.currents[0, 0, g]
				jminus = -self.coarse_mesh.currents[1, 1, g]
				dhatr = -(jplus + dsquigr*(phir - phic))/(phir + phic)
				# Matrix entries...TODO: Check the signs of these terms!
				c = (dsquigr + dhatr)/cmidx + node.sigma_tr[g]
				r = (dsquigr - dhatr)/cmidx
				#matA[i0, i0:(i0+2)] = [c, r]
			elif bcl == "vacuum":
				"""I'm 100% sure this is wrong."""
				dsquigr = self.dtildes[1, 1, g] # [1, 0, g]?
				# TODO: check indexing. Pretty sure this actually uses jminus?
				jplus = -self.coarse_mesh.currents[0, 0, g]
				jminus = -self.coarse_mesh.currents[1, 1, g]
				dhatr = -(jplus + dsquigr*(phir - phic))/(phir + phic)
				# Matrix entries
				c =  (dsquigr - dhatr - 0.5*jminus/phic)/cmidx + node.sigma_tr[g]
				r = -(dsquigr + dhatr)/cmidx
			else:
				raise NotImplementedError(bcl)
			matA[i0, i0:(i0+2)] = [c, r]
			
			node = self.coarse_mesh.nodes[-1]
			cmidx = node.dx
			if bcr == "reflective":
				phil = flux[-2, g]
				phic = flux[-1, g]
				dsquigl = self.dtildes[0, -1, g]  # [1, -2, g]?
				# TODO: determine whether this is J+ or J-
				jplus = -self.coarse_mesh.currents[0, -2, g]
				jminus = -self.coarse_mesh.currents[1, -1, g]
				dhatl = -(jminus + dsquigl*(phil - phic))/(phil + phic)
				# Matrix entries
				l = (dsquigl - dhatl)/cmidx
				c = (dsquigl + dhatl)/cmidx + node.sigma_tr[g]
			#elif bcr == "vacuum":
			else:
				raise NotImplementedError(bcr)
			matA[i0+NX-1, (i0+NX-2):(i0+NX)] = [l, c]
			
		return matA
	
	def _build_matrix_b(self, flux, k):
		G = self.coarse_mesh.groups
		NX = self.coarse_mesh.nx
		NC = NX*G
		matB = scipy.zeros(NC)
		for g in range(G):
			i0 = g*NX
			for cmi in range(NX):
				node = self.coarse_mesh.nodes[cmi]
				fsi = 0.0
				for gp in range(G):
					phip = flux[cmi, gp]
					# Fission source
					fsi += node.chi[g]*node.nu_sigma_f[gp]*phip/k
					# Scatter source
					fsi += node.scatter_matrix[g, gp]*phip
				matB[i0 + cmi] = fsi
		return matB
			
	def solve(self, ss, fs, k, eps):
		G = self.coarse_mesh.groups
		NX = self.coarse_mesh.nx
		NC = NX*G
		
		old_flux = self.coarse_mesh.flux
		
		fsdiff = 1
		kdiff = 1
		outer_count = 0
		oldS = self._build_matrix_b(old_flux, 1)
		oldk = k
		while ((fsdiff > 1E-5) or (kdiff > 1E-5)) and (outer_count < 1000):
			matA = self._build_matrix_a(old_flux)
			matB = self._build_matrix_b(old_flux, oldk)
			fluxdiff = 1
			inner_count = 0
			while fluxdiff > eps:
				flux = linalg.solve(matA, matB)
				flux.shape = (G, NX)
				flux = flux.T
				
				fluxdiff = 0
				for g in range(G):
					for i in range(NX):
						phi1 = flux[i, g]
						phi0 = old_flux[i, g]
						if phi1 != phi0:
							fluxdiff += ((phi1 - phi0)/phi1)**2
					fluxdiff = scipy.sqrt(fluxdiff/NC)
				
				old_flux = scipy.array(flux)
				inner_count += 1
				if inner_count >= 1000:
					raise SystemError("Maximum inner iterations reached")
				
			#print("Diffusion converged after {} inner iterations.".format(inner_count))
			# Now we've converged the flux at the fission source
			newS = self._build_matrix_b(flux, 1)
			k = oldk*newS.sum()/oldS.sum()
			kdiff = abs(k - oldk)/oldk
			
			outer_count += 1
			#print("Diffusion Outer {}: k = {}, kdiff = {}".format(outer_count, k, kdiff))
			
			fsdiff = 0
			for i in range(NC):
				fs1 = newS[i]
				fs0 = oldS[i]
				if fs1 != fs0:
					fsdiff+= ((fs1 - fs0)/fs1)**2
			fsdiff = scipy.sqrt(fsdiff/NC)
			
			oldS = scipy.array(newS)
			oldk = k
			
			if outer_count >= 1000:
				raise SystemError("Maximum inner iterations reached")
		
		print("Diffusion converged after {} outer iterations.".format(outer_count))
		self.coarse_mesh.flux = flux
	
	