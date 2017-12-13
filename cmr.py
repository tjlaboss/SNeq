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
					fsi += (fine_fs[i, g]/k + fine_ss[i, g])*dxi
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
			cmdxi = self.nodes[cmi].dx
			for g in range(self.groups):
				#fine_mesh.flux[i, g] *= fi[g]*(fine_mesh.nodes[i].dx/cmdxi)
				fine_mesh.flux[i, g] *= fi[g]#/cmdxi
		return fine_mesh.flux

