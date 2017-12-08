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
import numpy as np

FIXED_SOURCE = 0.0
G = 1

# Load the cross sections from disk
if G == 1:
	file1 = open(constants.FNAME1, "rb")
	mg = pickle.load(file1)
	file1.close()
elif G == 2:
	file2 = open(constants.FNAME2, "rb")
	mg = pickle.load(file2)
	file2.close()
else:
	raise NotImplementedError("{} groups".format(G))

mod_mat = material.Material(name="Moderator", groups=G)
mod_mat.macro_xs = mg["mod"]

fuel_mat = material.Material(name="Fuel, 3.1%", groups=G)
fuel_mat.macro_xs = mg["fuel"]

if G == 1:
	# debug
	fuel_mat.macro_xs["nu-fission"] = np.array([0.1])
	# analytically calculate kinf from the 1-group xs
	kinf = float(mg["fuel"]["nu-fission"]/mg["fuel"]["absorption"])
elif G == 2:
	# fudge the numbers for testing
	mg["fuel"]["nu-fission"][:] = 2.2*mg["fuel"]["absorption"][:]  # debug: force kinf to 2.2
	mg["fuel"]["nu-scatter"][0, 1] = 0.0 # no upscatter
	mg["mod"]["nu-scatter"][0, 1] = 0.0  # no upscatter
	
	f2 = mg["fuel"]
	sigma_s12 = f2["nu-scatter"][1, 0] - f2["nu-scatter"][0, 1]
	kinf = (f2["nu-fission"][0] + (sigma_s12/f2["absorption"][1])*f2["nu-fission"][1]) / (f2["absorption"][0] + sigma_s12)

print("kinf = {:1.5f}".format(kinf))


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
		# debug
		self.flux[:, :] = 1.0
		#self.update_nodal_fluxes()
	
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
	
	def calculate_fission_source(self):
		"""One-group. Temporary function. Delete this.
		
		Edit: just expanded to 2-group
		"""
		fs = np.zeros((self.nx, self.groups))
		for i in range(self.nx):
			node = self.nodes[i]
			if node.nu_sigma_f.any():
				for g in range(self.groups):
					if node.chi.any():
						chi = node.chi[g]
					else:
						# If no chi spectrum is required, it's 1 for the fast group
						chi = 1*(g==0)
					phi = self.flux[i, g]
					fs[i, g] = chi*phi*node.nu_sigma_f[g]
			else:
				fs[i, :] = 0.0
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
			if region == 1:
				fuel_node = node.Node1D(dx, self.quad, self.fuel.macro_xs, self.groups,
				                        source=FIXED_SOURCE)
				self.nodes[i] = fuel_node
			else:
				mod_node = node.Node1D(dx, self.quad, self.mod.macro_xs, self.groups)
				self.nodes[i] = mod_node


# test
s2 = quadrature.GaussLegendreQuadrature(6)
NXMOD = 0
NXFUEL = 6
KGUESS = kinf
cell = Pincell1D(s2, mod_mat, fuel_mat, nx_mod=NXMOD, nx_fuel=NXFUEL, groups=G)
solver = calculator.DiamondDifferenceCalculator1D(s2, cell, ("reflective", "reflective"), kguess=KGUESS)
#solver.transport_sweep(KGUESS)
solver.solve(eps=1E-6, maxiter=100)
phi = solver.mesh.flux
print(cell)
print(phi)
print(solver.k)
if True:
	if G == 1:
		plot1d.plot_1group_flux(cell, False, NXMOD)
	elif G == 2:
		plot1d.plot_2group_flux(cell, False, NXMOD)