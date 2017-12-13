# Mesh
#
# Classes and methods for discrete ordinate methods
# Designed to work for uniform meshes, or as constructor classes for
# non-uniform meshes

import node
import numpy as np
from constants import BOUNDARY_CONDITIONS

class Mesh(object):
	"""One-group, n-dimensional Cartesian mesh
	
	Parameters:
	-----------
	num:            int; number of dimensions
	quad:           Quadrature; num-D angular quadrature to use
	wxyz:           list(len=num) of floats, cm; the sizes of each dimension
	nxyz:           list(len=num) of ints; number of nodes of each dimension
	groups:         int; number of energy groups
	
	Attributes:
	-----------
	
	"""
	def __init__(self, num, quad, wxyz, nxyz, groups):
		assert len(wxyz) == num, \
			"Wrong number of entries for wxyz in {} dimensions".format(num)
		assert len(nxyz) == num, \
			"Wrong number of entries for nxyz in {} dimensions".format(num)
		self.num = num
		self.quad = quad
		#self.wxyz = wxyz
		#self.nxyz = nxyz
		self.groups = groups
		self.bcs = None
	
	def _populate(self):
		"""Populate the mesh with appropriate nodes"""
		pass
	
	def set_bcs(self, bcs):
		for bc in bcs:
			assert bc in BOUNDARY_CONDITIONS, \
				"{} is an unknown boundary condition.".format(bc)
		pass
		

class Mesh1D(Mesh):
	"""One-dimentional mesh
	
	Parameters:
	-----------
	quad:           Quadrature; 1-D angular quadrature to use
	

	Attributes:
	-----------
	
	"""
	def __init__(self, quad, xwidth, nx, groups):
		super().__init__(1, quad, [xwidth], [nx], groups)
		self.nx = nx
		self.xwidth = xwidth
		self.nodes = np.empty(nx, dtype=node.Node1D)
		self.flux = np.ones((nx, groups))
		self.psi = np.zeros((nx + 1, quad.N, groups))
		
	def get_dx(self, i):
		"""Return the mesh spacing for the ith node.
		
		As written, simply returns xwidth/nx for all cases.
		
		This method is intended to be overwritten by a custom function
		for non-uniform methods.
		"""
		return self.xwidth/self.nx
	
	def set_bcs(self, bcs):
		if bcs is not None:
			super().set_bcs(bcs)
			assert len(bcs) == 2, \
				"A 1D mesh requires 2 boundary conditions."
			self.bcs = bcs
	
	def update_nodal_fluxes(self):
		"""Update the scalar flux in the nodes from that on the mesh."""
		for g in range(self.groups):
			for i in range(self.nx):
				self.nodes[i].flux[g] = self.flux[i, g]

