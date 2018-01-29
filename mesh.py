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
	
	def calculate_scatter_source(self):
		ss = np.zeros((self.nx, self.groups))
		for i in range(self.nx):
			phi_vector = self.flux[i]
			ss[i] = self.nodes[i].scatter_matrix.dot(phi_vector)
		return ss
	

class Mesh2D(Mesh):
	"""Two-dimentional mesh

	Parameters:
	-----------
	quad:           Quadrature; 2-D or 3-D angular quadrature to use


	Attributes:
	-----------
	area:           float, cm^2; total area of the mesh
	"""
	def __init__(self, quad, xwidth, ywidth, nx, ny, groups):
		super().__init__(2, quad, [xwidth, ywidth], [nx, ny], groups)
		self.nx = nx
		self.ny = ny
		self.xwidth = xwidth
		self.ywidth = ywidth
		self.area = xwidth*ywidth
		self.nodes = np.empty((nx, ny), dtype=node.Node2D)
		self.flux = np.ones((nx, ny, groups))
		# Overall angular flux: Will eventually be removed.
		self.psi = np.zeros((nx + 1, ny, quad.Nflux, groups))
		# Boundary fluxes: here to stay.
		self.psi_north = np.zeros((nx, quad.Nflux, groups))
		self.psi_south = np.zeros((nx, quad.Nflux, groups))
		self.psi_west = np.zeros((ny, quad.Nflux, groups))
		self.psi_east = np.zeros((ny, quad.Nflux, groups))
	

	def calculate_scatter_source(self):
		ss = np.zeros((self.nx, self.ny, self.groups))
		for i in range(self.nx):
			for j in range(self.ny):
				phi_vector = self.flux[i, j]
				ss[i, j] = self.nodes[i, j].scatter_matrix.dot(phi_vector)
		return ss
	
