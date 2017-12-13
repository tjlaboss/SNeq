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
		self.coarse_mesh.prolong_flux(self.fine_mesh)
	
	def solve(self, eps):
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
			transport = node.sigma_tr[g]*phidx#*node.dx
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
	def __init__(self, coarse_mesh, fine_mesh, bcs):
		super().__init__(coarse_mesh, fine_mesh, bcs)
		
		# FIXME: This just does a generic diffusion solve, not actual CMFD.
		# TODO: Modify diffusion kernel to do proper CMFD.
		if coarse_mesh.nx_mod:
			print("This should not execute in homogeneous medium.")
			widths = scipy.array([
				coarse_mesh.mod_xwidth, coarse_mesh.fuel_xwidth, coarse_mesh.mod_xwidth])
			mats = scipy.array(
				[coarse_mesh.mod, coarse_mesh.fuel, coarse_mesh.mod])
			dxs = scipy.array(
				[coarse_mesh.dx_mod, coarse_mesh.dx_fuel, coarse_mesh.dx_mod])
		else:
			widths = scipy.array([coarse_mesh.fuel_xwidth])
			mats = scipy.array([coarse_mesh.fuel])
			dxs = scipy.array([coarse_mesh.dx_fuel])
		self.diffusion_problem = diffusion.Problem(dxs, widths, mats, bcs)
	
	def solve(self, flux, fs, k, eps):
		# TODO: Write the CMFD diffusion solver correctly
		flux, fs, k = diffusion.solve_problem(self.diffusion_problem,
		        fluxguess=flux, fsguess=fs, kguess=k,
		        eps_inner=eps, eps_outer=eps, plot=False)
		self.coarse_mesh.flux = flux
		# TODO: Figure out what to do with these
		return flux, fs, k
