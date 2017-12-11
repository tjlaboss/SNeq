# Accelerator
#
# There are various acceleration methods one can use with discrete ordinates
# radiation transport. The Accelerator and its derived classes serve as
# wrappers for acceleration methods of various colors and shapes.


class Accelerator(object):
	"""Base class for acceleration wrappers
	
	
	"""
	def __init__(self, coarse_mesh):
		self.coarse_mesh = coarse_mesh
		pass
	
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
	def __init__(self, coarse_mesh):
		super().__init__(coarse_mesh)
	
	def solve(self, eps):
		# TODO: Confirm whether we solve CMR by transport or diffusion
		pass

