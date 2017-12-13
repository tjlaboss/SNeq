# CMR
#
# Coarse Mesh Rebalance tools

from _coarse_mesh import CoarseMeshPincell1D
import numpy as np


class RebalancePincell1D(CoarseMeshPincell1D):
	"""A Coarse Mesh Rebalance formulation of the 1D pincell"""
	def fromFineMesh(self, fine_mesh, ratio):
		"""Create a RebalanceMesh1D using the transport mesh as a template.
		Or another coarse mesh, CMR is naturally multi-level.

		TODO: Allow mixed fuel/moderator nodes.

		Parameters:
		-----------
		fine_mesh:      Mesh1D to base the coarse mesh off
		ratio:          int; ratio of the number of nodes across the mesh
		"""
		super().fromFineMesh(fine_mesh, ratio)
		nx_fuel = fine_mesh.nx_fuel//ratio
		nx_mod = fine_mesh.nx_mod//ratio
		cm = RebalancePincell1D(fine_mesh.quad, fine_mesh.mod, fine_mesh.fuel,
		                        fine_mesh.pitch, fine_mesh.width, nx_mod, nx_fuel,
		                        fine_mesh.groups, fine_mesh.source, ratio)
		cm.set_bcs(fine_mesh.bcs)
		return cm
	
	def _restrict_cross_sections(self):
		"""Placeholder for cross section restriction

		This will integrate each macro_xs over the volume of each node
		and basically imitate the Pincell1D._populate() function.

		This will be written once mixed fuel/mod cells are enabled.
		"""
		pass
	
	def _get_rebalance_factors(self, new_flux):
		"""Find the rebalance factors for prolongation.

		Parameter:
		----------
		new_flux:   array of the scalar flux on the coarse mesh
					after the latest linear solution

		Returns:
		--------
		factors:    array of rebalance factors on the coarse mesh
		"""
		return new_flux/self.flux
	
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
				# Integrate over the fine mesh
				for fmi in range(self.ratio):
					i = self.ratio*cmi + fmi
					dxi = fine_mesh.nodes[i].dx
					for n in range(self.quad.N):
						psi_n0 = fine_mesh.psi[i, n, g]#*dxi     # LHS integrated flux
						psi_n1 = fine_mesh.psi[i+1, n, g]#*dxi # RHS integrated flux
						if n < self.quad.N2:
							jplus += self.quad.weights[n]*abs(self.quad.mus[n])*psi_n1
						else:
							jminus += self.quad.weights[n]*abs(self.quad.mus[n])*psi_n0
						#phi += 0.5*(psi_n0 + psi_n1)  # diamond difference approxmation
					# I wonder if this is more accurate?
					phi += fine_mesh.flux[i, g]#*dxi#??
				self.currents[0, cmi+1, g] = jplus
				self.currents[1, cmi, g] = jminus
				self.flux[cmi, g] = phi
			# Boundary conditions
			#print("Currents before BCs:")
			#print(self.currents[:,:,g])
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
			#print("Currents with BCs:")
			#print(self.currents[:, :, g])
			#print()
	
	def get_coarse_source(self, fine_mesh, fine_ss, fine_fs, k):
		"""Integrate the fine mesh fission source to get the
		coarse mesh source vector.

		Parameters:
		-----------
		fine_source:        multigroup array of the fission source
		k:                  float; latest guess of the eigenvalue
		"""
		coarse_source = np.zeros((self.nx*self.groups))
		index = lambda ii, gg: ii + self.nx*gg
		for g in range(self.groups):
			for cmi in range(self.nx):
				fsi = 0.0
				for fmi in range(self.ratio):
					i = self.ratio*cmi + fmi
					dxi = fine_mesh.nodes[i].dx
					fsi += (fine_fs[i, g]/k + fine_ss[i, g])#*dxi
				j = index(cmi, g)
				coarse_source[j] = fsi
		return coarse_source
	
	def prolong_flux(self, fine_mesh, factors):
		"""Use the coarse mesh flux to update the fine mesh flux.
		This RebalancePincell1D's flux will be updated to the
		coarse flux after the prolongation is complete.

		Parameter:
		----------
		fine_mesh:      Pincell1D instance
		coarse_flux:    array of the new coarse flux to use for the prolongation.
		"""
		for i in range(fine_mesh.nx):
			cmi = i//self.ratio
			fi = factors[cmi]
			for g in range(self.groups):
				fine_mesh.flux[i, g] *= fi[g]
		return fine_mesh.flux

