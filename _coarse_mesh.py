# Coarse Mesh
#
# Base classes and common functionality for CMR and CMFD

import pincell
import numpy as np

class CoarseMeshPincell1D(pincell.Pincell1D):
	"""Constructor class for coarse mesh acceleration methods"""
	def __init__(self, quad=None, mod=None, fuel=None, nx_mod=None, nx_fuel=None,
	             groups=None, source=None, ratio=None):
		if not ((quad is None) or (groups is None) or (ratio is None)):
			super().__init__(quad, mod, fuel, nx_mod, nx_fuel, groups, source)
			self.ratio = ratio
			self.currents = np.zeros((2, self.nx, self.groups))
			self._populate()
		self.psi = None
	
	def fromFineMesh(self, fine_mesh, ratio):
		"""Generate a coarse mesh for the accelerated method from
		the fine mesh for the tranport solution
		
		Parameters:
		-----------
		fine_mesh:      Mesh1D to base the coarse mesh off
		ratio:          int; ratio of the number of nodes across the mesh
		
		"""
		assert not fine_mesh.nx_fuel % ratio, \
			"Ratio does not produce an integer number of fuel nodes."
		assert not fine_mesh.nx_mod % ratio, \
			"Ratio does not produce an integer number of moderator nodes."
		pass

