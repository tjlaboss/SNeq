# Problem specification
#
# Constants, dimensions, etc. for the solver
# Expect this file to change greatly over the course of its development

import node
import mesh
import quadrature
import material
import calculator
import plot2d

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
u238 = material.Nuclide(238, {"nu-scatter": SIGMA_S_U238})
o16 = material.Nuclide(16, {"nu-scatter": SIGMA_S_O16})
h1 = material.Nuclide(1, {"nu-scatter": SIGMA_S_H1,
                          "absorption": SIGMA_A_H1})
# Define the constituent materials
fuel_mat = material.Material().fromNuclides([u238], RHO_FUEL, name="Fuel")
fuel_mat.macro_xs["transport"] = sum(fuel_mat.macro_xs.values())
mod_mat = material.Material().fromNuclides([o16, h1, h1], RHO_MOD, name="Moderator")
#for xs in mod_mat.macro_xs:
#	mod_mat.macro_xs[xs] = 0.5*fuel_mat.macro_xs[xs]
mod_mat.macro_xs["transport"] = sum(mod_mat.macro_xs.values())

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
		# X-Ranges
		self.mod0_xlim0 = 0
		self.mod0_xlim1 = self.nx_mod - 1
		self.fuel_xlim0 = self.nx_mod
		self.fuel_xlim1 = self.nx_mod + self.nx_fuel - 1
		self.mod1_xlim0 = self.nx_mod + self.nx_fuel
		self.mod1_xlim1 = self.nx - 1
		# Y-Ranges
		self.mod0_ylim0 = 0
		self.mod0_ylim1 = self.ny_mod - 1
		self.fuel_ylim0 = self.ny_mod
		self.fuel_ylim1 = self.ny_mod + self.ny_fuel - 1
		self.mod1_ylim0 = self.ny_mod + self.ny_fuel
		self.mod1_ylim1 = self.nx - 1
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
	[{self.mod0_xlim0}, {self.mod0_xlim1}]: Moderator
	[{self.fuel_xlim0}, {self.fuel_xlim1}]: Fuel
	[{self.mod1_xlim0}, {self.mod1_xlim1}]: Moderator
""".format(**locals())
		return rep
	
	def get_region(self, i, j):
		if self.mod0_xlim1 < i < self.mod1_xlim0 and self.mod0_ylim1 < j < self.mod1_ylim0:
			return 1, 1
		elif self.mod0_xlim1 < i < self.mod1_xlim0:
			return 1, 0
		elif self.mod0_ylim1 < j < self.mod1_ylim0:
			return 0, 1
		else:
			return 0, 0
	
	def get_dxy(self, i, j):
		kx, ky = self.get_region(i, j)
		return self._dxs[kx], self._dys[ky]
	
	def _populate(self):
		for i in range(self.nx):
			for j in range(self.ny):
				region = self.get_region(i, j)
				dx, dy = self.get_dxy(i, j)
				if region == (1, 1):
					fuel_node = node.Node2D(dx, dy, self.quad, self.fuel.macro_xs,
					                        self.groups, FIXED_SOURCE, name="Fuel")
					self.nodes[i, j] = fuel_node
				else:
					mod_node = node.Node2D(dx, dy, self.quad, self.mod.macro_xs,
					                       self.groups, name="Moderator")
					self.nodes[i, j] = mod_node
			
	def calculate_fission_source(self):
		return None

# test
#BOUNDARIES = ["vacuum"]*4
BOUNDARIES = ["periodic"]*4
#BOUNDARIES = ["reflective"]*2 + ["periodic"]*2
#BOUNDARIES = ["periodic"]*2 + ["reflective"]*2
#BOUNDARIES = ["reflective"]*4
NFUEL = 12*3
NMOD = 4*3
s4 = quadrature.LevelSymmetricQuadrature2D(16)
cell = Pincell2D(s4, mod_mat, fuel_mat, NMOD, NFUEL, NMOD, NFUEL)
print(cell._dxs)

import numpy as np
ntot = NFUEL+NMOD
q_over_sigma = np.empty((NFUEL+NMOD, NFUEL+NMOD))
for i in range(ntot):
	for j in range(ntot):
		node = cell.nodes[i, j]
		q_over_sigma[i, j] = node.source / node.sigma_tr[0]

solver = calculator.DiamondDifferenceCalculator2D(s4, cell, BOUNDARIES, kguess=None)
#solver.transport_sweep(False)
import time
t1 = time.clock()
converged = solver.solve(eps=1E-5)
t2 = time.clock()
phi = solver.mesh.flux[:,:,0]
print(phi)
print(t2 - t1, "seconds") #8.754909
converged=True
if converged:
	plot2d.plot_1group_flux(cell, True, nxmod=5)
