# CMFD
#
# Coarse Mesh Finite Difference acceleration classes

from _coarse_mesh import CoarseMeshPincell1D
import diffusion
import numpy as np

class FiniteDifferencePincell1D(CoarseMeshPincell1D):
	"""A CMFD formulation of the 1D pincell"""
	def __init__(self, quad=None, mod=None, fuel=None,
	             pitch=0, width=0, nx_mod=0, nx_fuel=0,
	             groups=None, source=None, ratio=None):
		
		if not ((quad is None) or (groups is None) or (ratio is None)):
			super().__init__(quad, mod, fuel, pitch, width, nx_mod, nx_fuel, groups, source, ratio)

	def fromFineMesh(self, fine_mesh, ratio):
		"""Generate a coarse mesh for the accelerated method from
		the fine mesh for the tranport solution
	
		Parameters:
		-----------
		fine_mesh:      Mesh1D to base the coarse mesh off
		ratio:          int; ratio of the number of nodes across the mesh

		Returns:
		--------
		FiniteDifferencePincell1D
		"""
		super().fromFineMesh(fine_mesh, ratio)
		nx_fuel = fine_mesh.nx_fuel//ratio
		nx_mod = fine_mesh.nx_mod//ratio
		cm = FiniteDifferencePincell1D(fine_mesh.quad,
		        fine_mesh.mod, fine_mesh.fuel,
				fine_mesh.pitch, fine_mesh.width, nx_mod, nx_fuel,
		        fine_mesh.groups, fine_mesh.source, ratio)
		cm.bcs = fine_mesh.bcs
		return cm
	
	def restrict_flux(self, fine_mesh):
		raise NotImplementedError("Must implement CMFD flux restriction before continuing.")
	
	def prolong_flux(self, fine_mesh, coarse_flux):
		raise NotImplementedError("Must implement CMFD flux prolongation before continuing.")
