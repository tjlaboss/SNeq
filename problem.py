# Problem specification
#
# Constants, dimensions, etc. for the solver
# Expect this file to change greatly over the course of its development

import node
import mesh
import quadrature
import material
import calculator
import plot2d, plot_angular

FIXED_SOURCE = 1.0

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
mod_mat.macro_xs["transport"] = sum(mod_mat.macro_xs.values())

# Cell dimensions
PITCH = 1.25            # cm; pin pitch
WIDTH = 0.80            # cm; length of one side of the square fuel pin
BOUNDARIES = ["periodic"]*4


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
		# Fuel-to-moderator ratio
		self._fm = -1
		
	
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
	
	def get_fm_flux_ratio(self):
		fuel_flux = 0.0
		mod_flux = 0.0
		for g in range(self.groups):
			for i in range(self.nx):
				for j in range(self.ny):
					area = self.nodes[i, j].area
					region = self.get_region(i, j)
					if region == (1, 1):
						fuel_flux += self.flux[i, j, g]*area
					else:
						mod_flux += self.flux[i, j, g]*area
		return fuel_flux/mod_flux
	
	def test_convergence(self, eps=0.001):
		"""Check whether the convergence criterion
		for this problem has been met.
		
		For PSet05, this is a 0.1% change in the scalar flux ratio
		between the fuel and the moderator.
		
		Returns:
		--------
		Boolean; True if converged, otherwise False
		"""
		new_ratio = self.get_fm_flux_ratio()
		if abs(new_ratio - self._fm)/self._fm > eps:
			converged = False
		else:
			converged = True
		self._fm = new_ratio
		return converged
	
'''
# Part 1
NFUEL = 4
NMOD = 1
s4 = quadrature.LevelSymmetricQuadrature2D(4)
report = "\nProblem 1:"
diff = 0.0
last_ratio = None
for n in range(6):
	#cell = Pincell2D(s4, mod_mat, fuel_mat, NMOD, NFUEL, NMOD, NFUEL)
	nmod = (n+1)*NMOD
	nfuel = (n+1)*NFUEL
	cell = Pincell2D(s4, mod_mat, fuel_mat, nmod, nfuel, nmod, nfuel)
	solver = calculator.DiamondDifferenceCalculator2D(s4, cell, BOUNDARIES, kguess=None)
	converged = solver.solve(eps=1E-6, test_convergence=cell.test_convergence)
	#
	ratio = cell.get_fm_flux_ratio()
	if n > 0:
		diff = abs(ratio - last_ratio)/ratio
	report += "\n\tMesh: \t|  Fuel-to-Moderator flux ratio: \t| % change: "
	report += "\n\t" + "-"*60
	report += "\n\t{}x{} \t|  {:.4f}                        \t| {:.3%}".format(2*nmod+nfuel, 2*nmod+nfuel, ratio, diff)
	dxf, dxm = cell._dxs[:2]
	report += '\n\t\tMesh size: {:.3f} cm fuel, {:.3f} cm mod'.format(dxf, dxm)
	report += "\n\t" + "="*79
	last_ratio = ratio
print(report)


# Part 2
NFUEL = 12
NMOD = 3
ns = (2, 4, 8, 16)
report = "\nProblem 2:"
report += "\n\tOrder:\t|  Fuel-to-Moderator flux ratio: "
report += "\n\t" + "-"*60
for n in ns:
	sn = quadrature.LevelSymmetricQuadrature2D(n)
	cell = Pincell2D(sn, mod_mat, fuel_mat, NMOD, NFUEL, NMOD, NFUEL)
	solver = calculator.DiamondDifferenceCalculator2D(sn, cell, BOUNDARIES, kguess=None)
	converged = solver.solve(eps=1E-6, test_convergence=cell.test_convergence)
	ratio = cell.get_fm_flux_ratio()
	order = str(n).ljust(2)
	report += "\n\t S{}\t|  {:.4f}".format(order, ratio)
report += "\n" + "="*79
print(report)

#if converged:
#	plot2d.plot_1group_flux(cell, True, nxmod=5, grid=False)
'''

# Part 3
NFUEL = 12
NMOD = 3
s4 = quadrature.LevelSymmetricQuadrature2D(4)
cell = Pincell2D(s4, mod_mat, fuel_mat, NMOD, NFUEL, NMOD, NFUEL)
solver = calculator.DiamondDifferenceCalculator2D(s4, cell, BOUNDARIES, kguess=None)
converged = solver.solve(eps=1E-3, test_convergence=cell.test_convergence)

fcenter = NMOD + NFUEL // 2
fedge = NMOD + NFUEL
cedge = 2*NMOD + NFUEL
cell.psi /= cell.psi.mean()
plot_angular.plot_1group_angular_flux(cell, fcenter, fcenter, "Center of fuel")
plot_angular.plot_1group_angular_flux(cell, fedge, fcenter, "Corner of fuel")
plot_angular.plot_1group_angular_flux(cell, fedge, fedge, "Edge of fuel")
plot_angular.plot_1group_angular_flux(cell, cedge, cedge, "Edge of cell")


import pylab
pylab.show()
