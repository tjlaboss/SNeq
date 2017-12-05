# Problem specification
#
# Constants, dimensions, etc. for the solver
# Expect this file to change greatly over the course of its development

import node
import mesh
import quadrature
import material
import calculator
from numpy import array

FIXED_SOURCE = 1.0  # TODO: scale by 1, 2, 4pi?


# Define the nuclides
# (Irrelevant, because manually overwritten below)
u238 = material.Nuclide(238, {"scatter": 10.0})
h1 = material.Nuclide(1, {"scatter"   : 1.0,
                          "absorption": 1.0})

# Define the constituent materials
fuel_mat = material.Material([u238], density=1.0, name="Fuel")
fuel_mat.macro_xs = {'scatter': array([10.0])}

mod_mat = material.Material([h1], density=1.0, name="Moderator")
mod_mat.macro_xs = {'absorption': array([1.0]),
                     'scatter': array([1.0])}

# Cell dimensions
PITCH = 0.6  # cm; pin pitch
WIDTH = 0.4  # cm; length of one side of the square fuel pin


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
	def __init__(self, quad, mod, fuel, nx_mod, nx_fuel, groups=1):
		nx = 2*nx_mod + nx_fuel
		super().__init__(quad, PITCH, nx, groups)
		self.fuel = fuel
		self.mod = mod
		assert groups == fuel_mat.g == mod_mat.g, \
			"Unequal number of energy groups in problem and materials."
		self.nx_mod = nx_mod
		self.nx_fuel = nx_fuel
		self.fuel_xwidth = WIDTH
		self.mod_xwidth = (PITCH - WIDTH)/2.0
		# Ranges
		self.mod0_lim0 = 0
		self.mod0_lim1 = self.nx_mod - 1
		self.fuel_lim0 = self.nx_mod
		self.fuel_lim1 = self.nx_mod + self.nx_fuel - 1
		self.mod1_lim0 = self.nx_mod + self.nx_fuel
		self.mod1_lim1 = self.nx - 1
		# Remove the next two lines for non-uniform meshes in each region
		self._dx_fuel = self.fuel_xwidth/self.nx_fuel
		if self.nx_mod:
			self._dx_mod = self.mod_xwidth/self.nx_mod
		else:
			self._dx_mod = 0.0
		self._dxs = (self._dx_mod, self._dx_fuel, self._dx_mod)
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
		return self._dxs[j]
	
	def _populate(self):
		for i in range(self.nx):
			region = self.get_region(i)
			dx = self.get_dx(i)
			if region == 1:
				fuel_node = node.Node1D(dx, self.quad, self.fuel.macro_xs, FIXED_SOURCE)
				self.nodes[i] = fuel_node
			else:
				mod_node = node.Node1D(dx, self.quad, self.mod.macro_xs)
				self.nodes[i] = mod_node
		# Initialize flux guess
		#for g in range(self.groups):
		#	for i in range(self.nx):
		#		self.flux[i, g] = self.nodes[i].flux[g]


# test
s2 = quadrature.GaussLegendreQuadrature(2)
NXMOD = 6
NXFUEL = 8
cell = Pincell1D(s2, mod_mat, fuel_mat, nx_mod=NXMOD, nx_fuel=NXFUEL)
solver = calculator.DiamondDifferenceCalculator1D(s2, cell, ("reflective", "reflective"))
solver.transport_sweep()
solver.solve(eps=1E-10)
phi = solver.mesh.flux
phi /= phi.max()
print(cell)
print(phi)
if True:
	from pylab import *
	xvals = range(0, cell.nx)
	plot(xvals, phi, "b-o")
	m = 1.1 #phi.max()
	plot([NXMOD, NXMOD], [0, m], "gray")
	plot([cell.nx-NXMOD-1, cell.nx-NXMOD-1], [0, m], "gray")
	xlabel("Node number", fontsize=14)
	ylabel("$\overline{\phi(x)}$", fontsize=14)
	titstr = "$S_" + str(s2.N) + "$, slab"
	title(titstr, fontsize=18)
	ylim([0, 1.1])
	grid()
	show()