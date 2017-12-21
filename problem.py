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

FIXED_SOURCE = 1.0  # TODO: scale by 1, 2, 4pi?

RHO_FUEL = 10.4     # g/cm^3
RHO_MOD = 0.7       # g/cm^3

# One-group cross sections
SIGMA_S_U238 = 11.29    # b
SIGMA_S_O16 = 3.888     # b
SIGMA_S_H1 = 20.47      # b
SIGMA_A_H1 = 1.0        # b
# --> fuel is pure scattering; only absorption is hydrogen

# Define the nuclides
u238 = material.Nuclide(238, {"scatter": SIGMA_S_U238})
o16 = material.Nuclide(16, {"scatter": SIGMA_S_O16})
h1 = material.Nuclide(1, {"scatter": SIGMA_S_H1,
                          "absorption": SIGMA_A_H1})
# Define the constituent materials
fuel_mat = material.Material().fromNuclides([u238], RHO_FUEL, name="Fuel")
mod_mat = material.Material().fromNuclides([o16, h1, h1], RHO_MOD, name="Moderator")


# Cell dimensions
PITCH = 1.25            # cm; pin pitch
WIDTH = 0.80            # cm; length of one side of the square fuel pin


class Pincell2D(mesh.Mesh2D):
	"""Mesh for a two-dimensional pincell with 3 regions:
	mod, fuel, and mod
	
	Parameters:
	-----------
	
	nx_mod:         int; number of mesh divisions in each moderator region
	nx_fuel:        int; number of mesh divisions in the one fuel region
	
	Attributes:
	-----------
	
	"""
	def __init__(self, quad, mod, fuel, nx_mod, nx_fuel, ny_mod, ny_fuel, groups=1):
		nx = 2*nx_mod + nx_fuel
		ny = 2*ny_mod + ny_fuel
		super().__init__(quad, PITCH, PITCH, nx, ny, groups)
		self.fuel = fuel
		self.mod = mod
		assert groups == fuel_mat.groups == mod_mat.groups, \
			"Unequal number of energy groups in problem and materials."
		self.nx_mod = nx_mod
		self.nx_fuel = nx_fuel
		self.ny_mod = ny_mod
		self.ny_fuel = ny_fuel
		self.fuel_xwidth = WIDTH
		self.mod_xwidth = (PITCH - WIDTH)/2.0
		self.fuel_ywidth = WIDTH
		self.mod_ywidth = (PITCH - WIDTH)/2.0
		# Ranges
		self.mod0_lim0 = 0
		self.mod0_lim1 = self.nx_mod - 1
		self.fuel_lim0 = self.nx_mod
		self.fuel_lim1 = self.nx_mod + self.nx_fuel - 1
		self.mod1_lim0 = self.nx_mod + self.nx_fuel
		self.mod1_lim1 = self.nx - 1
		# Remove the next four lines for non-uniform meshes in each region
		self._dx_fuel = self.fuel_xwidth/self.nx_fuel
		self._dy_fuel = self.fuel_ywidth/self.ny_fuel
		if self.nx_mod:
			self._dx_mod = self.mod_xwidth/self.nx_mod
			self._dy_mod = self.mod_ywidth/self.ny_mod
		else:
			self._dx_mod = 0.0
			self._dy_mod = 0.0
		self._dxs = (self._dx_mod, self._dx_fuel, self._dx_mod)
		self._dys = (self._dy_mod, self._dy_fuel, self._dy_mod)
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
	
	def get_region(self, i, j):
		if i <= self.mod0_lim1 or j <= self.mod0_lim0:
			return 0
		elif i <= self.fuel_lim1 and j <= self.fuel_lim1:
			return 1
		elif i <= self.mod1_lim1 or j <= self.mod1_lim1:
			return 2
		else:
			errstr = "Invalid index {} for mesh of size {}.".format(i, self.nx)
			raise IndexError(errstr)
	
	def get_dxy(self, i, j):
		k = self.get_region(i, j)
		return self._dxs[k], self._dys[k]
	
	def _populate(self):
		for i in range(self.nx):
			for j in range(self.ny):
				region = self.get_region(i, j)
				dx, dy = self.get_dxy(i, j)
				if region == 1:
					fuel_node = node.Node2D(dx, dy, self.quad, self.fuel.macro_xs, self.groups, FIXED_SOURCE)
					self.nodes[i] = fuel_node
				else:
					mod_node = node.Node2D(dx, dy, self.quad, self.mod.macro_xs, self.groups)
					self.nodes[i] = mod_node
			
	def calculate_fission_source(self):
		return None

# test
BOUNDARIES = ["vacuum"]*4
NFUEL = 8
NMOD = 0#5
s4 = quadrature.LevelSymmetricQuadrature(4)
cell = Pincell2D(s4, mod_mat, fuel_mat, NMOD, NFUEL, NMOD, NFUEL)
solver = calculator.DiamondDifferenceCalculator2D(s4, cell, BOUNDARIES, kguess=None)
solver.transport_sweep(None)
raise SystemExit
#solver.solve(eps=1E-10)
phi = solver.mesh.flux
print(phi)
print(mod_mat.macro_xs)
print(fuel_mat.macro_xs)
if True:
	plot1d.plot_1group_flux(cell, True, nxmod=5)
