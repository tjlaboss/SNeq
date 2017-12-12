# Pincell, 1D
#
# Class for a 1D pincell geometry, which we will be solving in this project

import mesh
import node
import numpy as np


class Pincell1D(mesh.Mesh1D):
	"""Mesh for a one-dimensional pincell with 3 regions:
	mod, fuel, and mod

	Parameters:
	-----------

	nx_mod:         int; number of mesh divisions in each moderator region
	nx_fuel:        int; number of mesh divisions in the one fuel region

	Attributes:
	-----------
	"""
	
	def __init__(self, quad, mod, fuel, pitch, width, nx_mod, nx_fuel,
	             groups=1, source=None):
		nx = 2*nx_mod + nx_fuel
		super().__init__(quad, pitch, nx, groups)
		self.source = source
		self.fuel = fuel
		self.mod = mod
		assert groups == fuel.groups == mod.groups, \
			"Unequal number of energy groups in problem and materials."
		self.pitch = pitch
		self.width = width
		self.nx_mod = nx_mod
		self.nx_fuel = nx_fuel
		self.fuel_xwidth = width
		self.mod_xwidth = (pitch - width)/2.0
		# Ranges
		self.mod0_lim0 = 0
		self.mod0_lim1 = self.nx_mod - 1
		self.fuel_lim0 = self.nx_mod
		self.fuel_lim1 = self.nx_mod + self.nx_fuel - 1
		self.mod1_lim0 = self.nx_mod + self.nx_fuel
		self.mod1_lim1 = self.nx - 1
		
		self.dxs = np.empty(self.nx)
		# Remove the next two lines for non-uniform meshes in each region
		self.dx_fuel = self.fuel_xwidth/self.nx_fuel
		if self.nx_mod:
			self.dx_mod = self.mod_xwidth/self.nx_mod
		else:
			self.dx_mod = 0.0
		self._dx_order = (self.dx_mod, self.dx_fuel, self.dx_mod)
		#
		self._populate()
	
	def __str__(self):
		rep = """\
1-D Pincell Mesh
----------------
Indices:
	[{self.mod0_lim0}, {self.mod0_lim1}]: Moderator
	[{self.fuel_lim0}, {self.fuel_lim1}]: Fuel
	[{self.mod1_lim0}, {self.mod1_lim1}]: Moderator
""".format(**locals())
		return rep
	
	def get_region(self, i):
		if i <= self.mod0_lim1:
			return 0
		elif i <= self.fuel_lim1:
			return 1
		elif i <= self.mod1_lim1:
			return 2
		else:
			errstr = "Invalid index {} for mesh of size {}.".format(i, self.nx)
			raise IndexError(errstr)
	
	def get_dx(self, i):
		j = self.get_region(i)
		return self._dx_order[j]
	
	def calculate_fission_source(self):
		"""Get the multigroup fission source array.

		Returns:
		--------
		fs:         array(nx, G) of the fission source
		"""
		fs = np.zeros((self.nx, self.groups))
		for i in range(self.nx):
			node = self.nodes[i]
			for g in range(self.groups):
				for gp in range(self.groups):
					fs[i, g] += node.chi[g]*self.flux[i, gp]*node.nu_sigma_f[gp]
		return fs
	
	def calculate_scatter_source(self):
		"""One-group. Temporary function. delete this too

		Edit: just expanded him to multigroup too
		"""
		ss = np.zeros((self.nx, self.groups))
		for i in range(self.nx):
			phi_vector = self.flux[i]
			ss[i] = self.nodes[i].scatter_matrix.dot(phi_vector)
		return ss
	
	def _populate(self):
		for i in range(self.nx):
			region = self.get_region(i)
			dx = self.get_dx(i)
			self.dxs[i] = dx
			if region == 1:
				fuel_node = node.Node1D(dx, self.quad, self.fuel, self.source)
				self.nodes[i] = fuel_node
			else:
				mod_node = node.Node1D(dx, self.quad, self.mod)
				self.nodes[i] = mod_node

