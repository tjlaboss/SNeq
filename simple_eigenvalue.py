# Problem specification
#
# Constants, dimensions, etc. for the solver
# Expect this file to change greatly over the course of its development

import node
import mesh
import quadrature
import material
import calculator
import plot1d
import constants
import pickle

FIXED_SOURCE = 1.0

# Load the cross sections from disk
file1 = open(constants.FNAME1, "rb")
mg1 = pickle.load(file1)
file1.close()
mod_mat = material.Material(name="Moderator", groups=1)
mod_mat.macro_xs = mg1["mod"]

fuel_mat = material.Material(name="Fuel, 3.1%", groups=1)
fuel_mat.macro_xs = mg1["fuel"]

'''
# analytically calculate kinf from the 1-group xs
# note: don't have absorption cross section...oops
kinf = mg1["fuel"]["nu-fission"]/(mg1["fuel"]["total"] - mg1["fuel"]["nu-scatter"])
kinf = float(kinf)
print("kinf = {:1.5f}".format(kinf))
exit()
'''


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
		assert groups == fuel_mat.groups == mod_mat.groups, \
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
				fuel_node = node.Node1D(dx, self.quad, self.fuel.macro_xs,
				                        source=FIXED_SOURCE)
				self.nodes[i] = fuel_node
			else:
				mod_node = node.Node1D(dx, self.quad, self.mod.macro_xs)
				self.nodes[i] = mod_node


# test
s2 = quadrature.GaussLegendreQuadrature(2)
NXMOD = 0
NXFUEL = 8
cell = Pincell1D(s2, mod_mat, fuel_mat, nx_mod=NXMOD, nx_fuel=NXFUEL)
solver = calculator.DiamondDifferenceCalculator1D(s2, cell, ("reflective", "reflective"), kguess=11)
solver.transport_sweep(11)
solver.solve(eps=1E-5, maxiter=1000)
phi = solver.mesh.flux
print(cell)
print(phi)
if False:
	plot1d.plot_1group_flux(cell, False, NXMOD)