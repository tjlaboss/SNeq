# CMFD
#
# Coarse Mesh Finite Difference acceleration classes

from _coarse_mesh import CoarseMeshPincell1D
import numpy as np

class FiniteDifferencePincell1D(CoarseMeshPincell1D):
	"""A CMFD formulation of the 1D pincell"""
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
		        fine_mesh.mod, fine_mesh.fuel, nx_mod, nx_fuel, fine_mesh.groups, ratio)
		return cm

