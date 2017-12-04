# Mesh
#
# Classes and methods for discrete ordinate methods
# Designed to work for uniform meshes, or as constructor classes for
# non-uniform meshes

import node
import numpy as np

class Mesh(object):
	"""One-group, n-dimensional Cartesian mesh
	
	Parameters:
	-----------
	num:            int; number of dimensions
	quad:           Quadrature; num-D angular quadrature to use
	wxyz:           list(len=num) of floats, cm; the sizes of each dimension
	nxyz:           list(len=num) of ints; number of nodes of each dimension
	
	Attributes:
	-----------
	
	"""
	def __init__(self, num, quad, wxyz, nxyz):
		assert len(wxyz) == num, \
			"Wrong number of entries for wxyz in {} dimensions".format(num)
		assert len(nxyz) == num, \
			"Wrong number of entries for nxyz in {} dimensions".format(num)
		self.num = num
		self.quad = quad
		#self.wxyz = wxyz
		#self.nxyz = nxyz
	
	def _populate(self):
		"""Populate the mesh with appropriate nodes"""
		pass


class Mesh1D(Mesh):
	"""One-dimentional mesh
	
	Parameters:
	-----------
	quad:           Quadrature; 1-D angular quadrature to use
	

	Attributes:
	-----------
	
	"""
	def __init__(self, quad, xwidth, nx):
		super().__init__(1, quad, [xwidth], [nx])
		self.nx = nx
		self.xwidth = xwidth
		self.nodes = np.empty(nx, dtype=node.Node1D)
		self.flux = np.zeros(nx)
		self.psi = np.zeros((nx, quad.N))
		
	def get_dx(self, i):
		"""Return the mesh spacing for the ith node.
		
		As written, simply returns xwidth/nx for all cases.
		
		This method is intended to be overwritten by a custom function
		for non-uniform methods.
		"""
		return self.xwidth/self.nx
	
	def update_nodal_fluxes(self):
		"""Update the scalar flux in the nodes from that on the mesh."""
		for i in range(self.nx):
			self.nodes[i].flux = self.flux[i]

