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
G = 2

# Load the cross sections from disk
if G == 1:
	file1 = open(constants.FNAME1, "rb")
	mg = pickle.load(file1)
	file1.close()
elif G == 2:
	# file2 = open(constants.FNAME2, "rb")
	# mg = pickle.load(file2)
	# file2.close()

	# 3.1% enriched fuel from 22.211/PSet06
	mg = {"fuel": {}, "mod": {}}
	fuel_diffusion = np.array([1.43, 0.37])
	mg["fuel"]["D"] = fuel_diffusion
	mg["fuel"]["transport"] = 1/(3*fuel_diffusion)
	mg["fuel"]["absorption"] = np.array([0.0088, 0.0852]) # 3%
	mg["fuel"]["nu-fission"] = np.array([0.0062, 0.1249])
	
	s12 = 0.0188
	scatter = mg["fuel"]["transport"] - mg["fuel"]["absorption"]
	s11 = scatter[0] - s12
	s22 = scatter[1]
	fuel_scatter_matrix = np.array([[s11,    0],
	                                [s12,  s22]])
	mg["fuel"]["nu-scatter"] = fuel_scatter_matrix
	
	
	# Water reflector
	mod_diffusion = np.array([1.55, 0.27])
	mg["mod"]["D"] = mod_diffusion
	mg["mod"]["transport"] = 1/(3*mod_diffusion)
	mg["mod"]["absorption"] = np.array([0.0010, 0.0300])
	
	s11, s22 = mg["mod"]["transport"] - mg["mod"]["absorption"]
	s12 = 0.0500
	mod_scatter_matrix = np.array([[s11, 0],
	                               [s12, s22]])
	mg["mod"]["nu-scatter"] = mod_scatter_matrix
	scatter = mod_scatter_matrix.sum(axis=0)
	mg["mod"]["total"] = mg["mod"]["transport"]
	
else:
	raise NotImplementedError("{} groups".format(G))

mod_mat = material.Material(name="Moderator", groups=G)
mod_mat.macro_xs = mg["mod"]

fuel_mat = material.Material(name="Fuel, 3.1%", groups=G)
fuel_mat.macro_xs = mg["fuel"]

if G == 1:
	# debug
	fuel_mat.macro_xs["nu-fission"] = 2.1*fuel_mat.macro_xs["absorption"] # debug; force kinf to 2.1
	fuel_mat.macro_xs["total"] = fuel_mat.macro_xs["transport"]
	# analytically calculate kinf from the 1-group xs
	kinf = float(mg["fuel"]["nu-fission"]/mg["fuel"]["absorption"])
elif G == 2:
	# fudge the numbers for testing
	'''
	mg["fuel"]["nu-fission"][:] = 2.2*mg["fuel"]["absorption"][:]  # debug: force kinf to 2.2
	mg["fuel"]["nu-scatter"][0, 1] = 0.0 # no upscatter
	mg["mod"]["nu-scatter"][0, 1] = 0.0  # no upscatter
	'''
	f2 = mg["fuel"]
	sigma_s12 = f2["nu-scatter"][1, 0] - f2["nu-scatter"][0, 1]
	numer = f2["nu-fission"][0] + (sigma_s12/f2["absorption"][1])*f2["nu-fission"][1]
	denom = sigma_s12 + f2["absorption"][0]
	#kinf = (f2["nu-fission"][0] + (sigma_s12/f2["absorption"][1])*f2["nu-fission"][1]) / (f2["absorption"][0] + sigma_s12)
	kinf = numer/denom

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
		'''
		for i in range(self.nx):
			node = self.nodes[i]
			if node.nu_sigma_f.any():
				if node.chi.any():
					chi = node.chi
				else:
					chi = np.zeros(self.groups)
					chi[0] = 1.0
				fs[i, :] = chi*self.flux[i, :]*node.nu_sigma_f
			else:
				fs[i, :] = 0.0
		'''
		# DEBUG; a messier fission source calculation just in case
		for i in range(self.nx):
			node = self.nodes[i]
			for g in range(self.groups):
				phi = self.flux[i, g]
				#print("g: {}, chi({}): {}".format(g, g, node.chi[g]))
				# FIXME: I think "chi" is what went wrong above!!
				fs[i, 0] += node.chi[0]*phi*node.nu_sigma_f[g]
				#fs[i, 1] += node.chi[1]*phi*node.nu_sigma_f[g]
		return fs
	
	def calculate_scatter_source(self):
		"""One-group. Temporary function. delete this too
		
		Edit: just expanded him to multigroup too
		"""
		ss = np.zeros((self.nx, self.groups))
		#'''
		for i in range(self.nx):
			phi_vector = self.flux[i]
			ss[i] = self.nodes[i].scatter_matrix.dot(phi_vector)
		'''
		# DEBUG; a messier scattering source calculation
		for i in range(self.nx):
			node = self.nodes[i]
			if G > 1:
				for g in range(self.groups):
					for gp in range(self.groups):
						phi = self.flux[i, gp]
						ss[i, g] += phi*node.scatter_matrix[g, gp]
			else:
				ss[i] = self.flux[i]*node.scatter_matrix
		'''
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


class RebalancePincell1D(Pincell1D):
	"""A Coarse Mesh Rebalance formulation of the 1D pincell"""
	def __init__(self, quad=None, mod=None, fuel=None,
	             nx_mod=None, nx_fuel=None, groups=None, ratio=None):
		if not ((quad is None) or (groups is None) or (ratio is None)):
			super().__init__(quad, mod, fuel, nx_mod, nx_fuel, groups)
			self.ratio = ratio
			self.currents = np.zeros((2, self.nx, self.groups))
			self._populate()
		self.psi = None
		
	def fromFineMesh(self, fine_mesh, ratio):
		"""Create a RebalanceMesh1D using the transport mesh as a template.
		Or another coarse mesh, CMR is naturally multi-level.

		TODO: Allow mixed fuel/moderator nodes.

		Parameters:
		-----------
		fine_mesh:      Mesh1D to base the coarse mesh off
		nx:             int; number of x-nodes in the coarse mesh
		"""
		assert not fine_mesh.nx % ratio, \
			"Ratio does not produce an integer number of fuel nodes."
		assert not fine_mesh.nx_mod % ratio, \
			"Ratio does not produce an integer number of moderator nodes."
		nx_fuel = fine_mesh.nx_fuel // ratio
		nx_mod = fine_mesh.nx_mod // ratio
		cm = RebalancePincell1D(fine_mesh.quad, fine_mesh.mod, fine_mesh.fuel,
		                        nx_mod, nx_fuel, fine_mesh.groups, ratio)
		return cm
	
	def _restrict_cross_sections(self):
		"""Placeholder for cross section restriction
		
		This will integrate each macro_xs over the volume of each node
		and basically imitate the Pincell1D._populate() function.
		
		This will be written once mixed fuel/mod cells are enabled.
		"""
		pass
	
	def _get_rebalance_factors(self, new_flux):
		"""Find the rebalance factors for prolongation.
		
		Parameter:
		----------
		new_flux:   array of the scalar flux on the coarse mesh
					after the latest linear solution
		
		Returns:
		--------
		factors:    array of rebalance factors on the coarse mesh
		"""
		return new_flux/self.flux
	
	def restrict_flux(self, fine_mesh):
		"""Integrate the angular flux from the transport solution over
		the coarse mesh cells. Use it to update	the coarse mesh scalar flux.
		
		Parameter:
		----------
		fine_mesh:      Pincell1D instance
		"""
		for g in range(self.groups):
			for cmi in range(self.nx):
				phi = 0.0
				jplus = 0.0
				jminus = 0.0
				# Integrate over the fine mesh
				for fmi in range(self.ratio):
					i = self.ratio*cmi + fmi
					dxi = fine_mesh.nodes[i].dx
					for n in range(self.quad.N):
						psi_n0 = fine_mesh.psi[i, n, g]#*dxi     # LHS integrated flux
						psi_n1 = fine_mesh.psi[i + 1, n, g]#*dxi # RHS integrated flux
						if n < self.quad.N2:
							jplus += self.quad.weights[n]*abs(self.quad.mus[n])*psi_n1
						else:
							jminus += self.quad.weights[n]*abs(self.quad.mus[n])*psi_n0
						phi += 0.5*(psi_n0 + psi_n1)  # diamond difference approxmation
				self.currents[:, cmi, g] = jplus, jminus
				self.flux[cmi, g] = phi
	
				
	def prolong_flux(self, fine_mesh, coarse_flux):
		"""Use the coarse mesh flux to update the fine mesh flux.
		This RebalancePincell1D's flux will be updated to the
		coarse flux after the prolongation is complete.
		
		Parameter:
		----------
		fine_mesh:      Pincell1D instance
		coarse_flux:    array of the new coarse flux to use for the prolongation.
		"""
		factors = self._get_rebalance_factors(coarse_flux)
		for i in range(fine_mesh.nx):
			cmi = i // self.ratio
			fi = factors[cmi]
			for g in range(self.groups):
				fine_mesh.flux[i, g] *= fi[g]
				# TODO: Confirm whether the angular flux should be likewise updated
				fine_mesh.psi[i, :, g] *= fi[g]
		self.flux = coarse_flux
		
	

# test fine mesh
s2 = quadrature.GaussLegendreQuadrature(2)
NXMOD = 0
NXFUEL = 8
KGUESS = kinf
cell = Pincell1D(s2, mod_mat, fuel_mat, nx_mod=NXMOD, nx_fuel=NXFUEL, groups=G)

solver = calculator.DiamondDifferenceCalculator1D(s2, cell, ("reflective", "reflective"), kguess=KGUESS)
solver.transport_sweep(KGUESS)

# test CMR
coarse_mesh = RebalancePincell1D().fromFineMesh(cell, 2)
coarse_mesh.restrict_flux(cell)
coarsine = np.array([np.cos((i-1.5)*np.pi/coarse_mesh.nx) for i in range(coarse_mesh.nx)])
new_flux = np.empty((coarse_mesh.nx, G))
new_flux[:, 0] = coarsine
new_flux[:, 1] = 0.5*coarsine
coarse_mesh.prolong_flux(cell, new_flux)

raise SystemExit

converged = solver.solve(eps=1E-6, maxiter=200)
phi = solver.mesh.flux
print(cell)
print(phi)
print(solver.k)
if converged:
	if G == 1:
		plot1d.plot_1group_flux(cell, True, NXMOD)
	elif G == 2:
		plot1d.plot_2group_flux(cell, True, NXMOD)