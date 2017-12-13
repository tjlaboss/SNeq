# Coarse Mesh
#
# Base classes and common functionality for CMR and CMFD

import pincell
import numpy as np

class CoarseMeshPincell1D(pincell.Pincell1D):
	"""Constructor class for coarse mesh acceleration methods"""
	def __init__(self, quad=None, mod=None, fuel=None,
	             pitch=0, width=0, nx_mod=0, nx_fuel=0,
	             groups=None, source=None, ratio=None):
		if not ((quad is None) or (groups is None) or (ratio is None)):
			super().__init__(quad, mod, fuel, pitch, width, nx_mod, nx_fuel, groups, source)
			self.ratio = ratio
			self.currents = np.zeros((2, self.nx+1, self.groups))
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
	
	def restrict_flux(self, fine_mesh):
		"""Integrate the angular flux from the transport solution over
		the coarse mesh cells. Use it to update	the coarse mesh scalar flux.

		Parameter:
		----------
		fine_mesh:      Pincell1D instance
		"""
		for g in range(self.groups):
			for cmi in range(self.nx):
				phi = 0.0
				jplus = 0.0
				jminus = 0.0
				il = cmi*self.ratio
				ir = il + self.ratio
				for n in range(self.quad.N):
					psi_n0 = fine_mesh.psi[il, n, g]  # *dxi  # LHS integrated flux
					psi_n1 = fine_mesh.psi[ir, n, g]  # *dxi  # RHS integrated flux
					if n < self.quad.N2:
						jplus += self.quad.weights[n]*abs(self.quad.mus[n])*psi_n1
					else:
						jminus += self.quad.weights[n]*abs(self.quad.mus[n])*psi_n0
				# Integrate over the fine mesh
				for fmi in range(self.ratio):
					i = self.ratio*cmi + fmi
					dxi = fine_mesh.nodes[i].dx
					'''
					for n in range(self.quad.N):
						psi_n0 = fine_mesh.psi[i, n, g]#*dxi     # LHS integrated flux
						psi_n1 = fine_mesh.psi[i+1, n, g]#*dxi # RHS integrated flux
						if n < self.quad.N2:
							jplus += self.quad.weights[n]*abs(self.quad.mus[n])*psi_n1
						else:
							jminus += self.quad.weights[n]*abs(self.quad.mus[n])*psi_n0
						#phi += 0.5*(psi_n0 + psi_n1)  # diamond difference approxmation
					'''
					# I wonder if this is more accurate?
					phi += fine_mesh.flux[i, g]#*dxi  # ??
				self.currents[0, cmi + 1, g] = jplus
				self.currents[1, cmi, g] = jminus
				self.flux[cmi, g] = phi
			# Boundary conditions
			# print("Currents before BCs:")
			# print(self.currents[:,:,g])
			bcl, bcr = self.bcs
			if bcl == "reflective":
				self.currents[0, 0, g] = self.currents[1, 0, g]
			elif bcl == "periodic":
				self.currents[0, 0, g] = self.currents[0, -1, g]
			elif bcl == "vacuum":
				# Should already be 0
				pass
			else:
				raise NotImplementedError(bcl)
			if bcr == "reflective":
				self.currents[1, -1, g] = self.currents[0, -1, g]
			elif bcr == "periodic":
				self.currents[1, -1, g] = self.currents[0, 0, g]
			elif bcr == "vacuum":
				# should already be 0
				pass
			else:
				raise NotImplementedError(bcr)
			# print("Currents with BCs:")
			# print(self.currents[:, :, g])
			# print()
	
	def prolong_flux(self, fine_mesh, coarse_flux):
		pass
