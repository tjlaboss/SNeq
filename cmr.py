# CMR
#
# Coarse Mesh Rebalance tools

from _coarse_mesh import CoarseMeshPincell1D


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
		                        nx_mod, nx_fuel, fine_mesh.groups, ratio)
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
						psi_n0 = fine_mesh.psi[i, n, g]  # *dxi     # LHS integrated flux
						psi_n1 = fine_mesh.psi[i + 1, n, g]  # *dxi # RHS integrated flux
						if n < self.quad.N2:
							jplus += self.quad.weights[n]*abs(self.quad.mus[n])*psi_n1
						else:
							jminus += self.quad.weights[n]*abs(self.quad.mus[n])*psi_n0
						phi += 0.5*(psi_n0 + psi_n1)  # diamond difference approxmation
				self.currents[:, cmi, g] = jplus, jminus
				self.flux[cmi, g] = phi
	
	def prolong_flux(self, fine_mesh, coarse_flux):
		"""Use the coarse mesh flux to update the fine mesh flux.
		This RebalancePincell1D's flux will be updated to the
		coarse flux after the prolongation is complete.

		Parameter:
		----------
		fine_mesh:      Pincell1D instance
		coarse_flux:    array of the new coarse flux to use for the prolongation.
		"""
		factors = self._get_rebalance_factors(coarse_flux)
		for i in range(fine_mesh.nx):
			cmi = i//self.ratio
			fi = factors[cmi]
			for g in range(self.groups):
				fine_mesh.flux[i, g] *= fi[g]
		self.flux = coarse_flux

