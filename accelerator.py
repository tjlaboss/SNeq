# Accelerator
#
# There are various acceleration methods one can use with discrete ordinates
# radiation transport. The Accelerator and its derived classes serve as
# wrappers for acceleration methods of various colors and shapes.

import diffusion
from constants import BOUNDARY_CONDITIONS
import numpy as np

class Accelerator(object):
	"""Base class for acceleration wrappers
	
	
	"""
	def __init__(self, coarse_mesh, bcs):
		self.coarse_mesh = coarse_mesh
		for bc in bcs:
			assert bc in BOUNDARY_CONDITIONS, \
				"{} is an unknown boundary condition.".format(bc)
		self.bcs = bcs
	
	def restrict(self, fine_mesh):
		"""Integrate the fine mesh onto the coarse mesh."""
		self.coarse_mesh.restrict_flux(fine_mesh)
	
	def prolong(self, fine_mesh, coarse_flux):
		"""Update the fine mesh using the coarse mesh."""
		self.coarse_mesh.prolong_flux(fine_mesh, coarse_flux)
	
	def solve(self, eps):
		"""Get the coarse mesh solution
		
		Parameter:
		----------
		eps:        float; relative tolerance
		"""
		pass


class RebalanceAccelerator1D(Accelerator):
	"""An Accelerator for the 1-D Coarse Mesh Rebalance method"""
	def __init__(self, coarse_mesh, bcs):
		super().__init__(coarse_mesh, bcs)
	
	def solve(self, eps):
		# TODO: Confirm whether we solve CMR by transport or diffusion
		pass


class FiniteDifference1D(Accelerator):
	"""An Accelerator for the 1-D CMFD method"""
	def __init__(self, coarse_mesh, bcs):
		super().__init__(coarse_mesh, bcs)
		
		# FIXME: This just does a generic diffusion solve, not actual CMFD.
		# TODO: Modify diffusion kernel to do proper CMFD.
		if coarse_mesh.nx_mod:
			print("This should not execute in homogeneous medium.")
			widths = np.array([
				coarse_mesh.mod_xwidth, coarse_mesh.fuel_xwidth, coarse_mesh.mod_xwidth])
			mats = np.array(
				[coarse_mesh.mod, coarse_mesh.fuel, coarse_mesh.mod])
			dxs = np.array(
				[coarse_mesh.dx_mod, coarse_mesh.dx_fuel, coarse_mesh.dx_mod])
		else:
			widths = np.array([coarse_mesh.fuel_xwidth])
			mats = np.array([coarse_mesh.fuel])
			dxs = np.array([coarse_mesh.dx_fuel])
		self.diffusion_problem = diffusion.Problem(dxs, widths, mats, bcs)
	
	def solve(self, flux, fs, k, eps):
		# TODO: Write the CMFD diffusion solver correctly
		flux, fs, k = diffusion.solve_problem(self.diffusion_problem,
		        fluxguess=flux, fsguess=fs, kguess=k,
		        eps_inner=eps, eps_outer=eps, plot=False)
		self.coarse_mesh.flux = flux
		# TODO: Figure out what to do with these
		return flux, fs, k
