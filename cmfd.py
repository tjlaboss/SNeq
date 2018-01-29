# CMFD
#
# Coarse Mesh Finite Difference acceleration classes

from _coarse_mesh import CoarseMeshPincell1D
import numpy as np

class FiniteDifferencePincell1D(CoarseMeshPincell1D):
	"""A CMFD formulation of the 1D pincell"""
	def __init__(self, quad=None, mod=None, fuel=None,
	             pitch=0, width=0, nx_mod=0, nx_fuel=0,
	             groups=None, source=None, ratio=None):
		
		if not ((quad is None) or (groups is None) or (ratio is None)):
			super().__init__(quad, mod, fuel, pitch, width, nx_mod, nx_fuel, groups, source, ratio)
			self._restricted_flux = None  # fine flux on coarse mesh before CMFD
		self.factor = None
			

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
		super(FiniteDifferencePincell1D, self).restrict_flux(fine_mesh)
		self._restricted_flux = np.array(self.flux)
	
	def prolong_flux(self, fine_mesh, coarse_flux):
		# TODO: Also update eigenvalue and boundary angular fluxes
		# debug: normalize flux ratio like this
		restricted_flux = self._restricted_flux/self._restricted_flux.mean()
		factors = self.flux/self.flux.mean()/restricted_flux
		self.factor = factors[len(factors)//2, 0]
		for i in range(fine_mesh.nx):
			cmi = i//self.ratio
			for g in range(self.groups):
				#fine_mesh.flux *= self.flux[cmi,g]/self._restricted_flux[cmi, g]
				fine_mesh.flux[i, g] *= factors[cmi, g]
		return fine_mesh.flux
