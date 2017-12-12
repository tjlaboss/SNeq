# Material
#
# Multi-group material classes

import constants
import numpy as np
from copy import deepcopy



class Nuclide(object):
	"""
	
	Parameters:
	-----------
	a:          float; mass number or atomic mass. you choose.
	xs_dict:    dict of {"reaction" : mgxs_list},
					-> mgxs_list: list of cross sections by energy group
	g:          int; number of energy groups
				[Default: 1]
	"""
	def __init__(self, a, xs_dict, g=1):
		self.a = a
		self.g = g
		self.xs_dict ={}
		for reaction, mgxs in xs_dict.items():
			assert reaction in constants.REACTIONS, \
				"Unknown reaction type: {}".format(reaction)
			if type(mgxs) in (int, float) and g == 1:
				mgxs = [mgxs]
			assert len(mgxs) == g,\
				"Wrong number of energy groups for {} xs".format(reaction)
			self.xs_dict[reaction] = np.array(mgxs)
		
		
class Material(object):
	"""
	
	Parameters:
	-----------
	macro_xs:       dict of {"reaction", array(macroscopic xs, cm^-1)}
	groups:         int; number of energy groups
	name:           str; descriptive name of the material
					[Default: empty string]
	
	Attributes:
	-----------
	g:              int; number of energy groups
	molar_mass:     float; mass of one mole of the material.
	number_density: float; number of atoms/cm^3 * 1E-24
	"""
	def __init__(self, macro_xs = None, groups=None, name=""):
		self.groups = groups
		self.name = name
		if macro_xs is not None:
			self.sigma_a, self.scatter_matrix, self.nu_sigma_f, \
				self.chi, self.sigma_tr, self.D = \
				self._group_cross_sections_from_dict(macro_xs)
	
	def __str__(self):
		return self.name
	
	def _group_cross_sections_from_dict(self, macro_xs):
		"""Read cross sections from a dictionary.

		Parameter:
		----------
		cross_sections:     dict of {"reaction" : macro_xs}

		Returns:
		--------
		sigma_a:            float, cm^-1; "absorption" xs
		sigma_s:            float, cm^-1; "scatter" xs
		nu_sigma_f:         float, cm^-1; "nu-fission" xs
		sigma_tr:           float, cm^-1;  transport xs
		"""
		sigma_a = np.zeros(self.groups)
		scatter_matrix = np.zeros((self.groups, self.groups))
		nu_sigma_f = np.zeros(self.groups)
		chi = np.zeros(self.groups)
		sigma_tr = np.zeros(self.groups)
		D = np.zeros(self.groups)
		
		if "absorption" in macro_xs:
			sigma_a = macro_xs["absorption"]
		if "nu-scatter" in macro_xs:
			scatter_matrix = macro_xs["nu-scatter"].squeeze()
		elif "scatter" in macro_xs:
			scatter_matrix = macro_xs["scatter"]
		if "nu-fission" in macro_xs:
			nu_sigma_f = macro_xs["nu-fission"]
		if "chi" in macro_xs:
			chi = macro_xs["chi"]
		elif nu_sigma_f.any():
			chi[0] = 1.0
		if "transport" in macro_xs:
			sigma_tr = macro_xs["transport"]
			D = 1.0/(3*sigma_tr)
		elif "D" in macro_xs:
			D = macro_xs["D"]
			sigma_tr = 1.0/(3*D)
		else:
			print("Warning: Transport cross section unavailable; using Total.")
			if "total" in macro_xs:
				sigma_tr = macro_xs["total"]
			else:
				sigma_tr = sigma_a  + scatter_matrix.sum(axis=0)
		return sigma_a, scatter_matrix, nu_sigma_f, chi, sigma_tr, D
	
	def fromNuclides(self, nuclides, density, name=""):
		"""Generate a Material object from a list of nuclides.
		Each Nuclide must have microscopic multigroup cross sections.
		Macroscopic multigroup cross sections will be calculated.
		
		Parameters:
		-----------
		nuclides:       list of Nuclides in the Material
		density:        float, g/cm^3: mass density of the material
		name:           str; descriptive name of the material
						[Default: empty string]
		
		Returns:
		--------
		Material.
		"""
		G = nuclides[0].g
		micro_xs = {}
		
		mgxs_temp = np.zeros(G)
		for reaction in constants.REACTIONS:
			micro_xs[reaction] = deepcopy(mgxs_temp)
		molar_mass = 0.0
		for nuc in nuclides:
			assert nuc.g == G, \
				"Nuclide {} has the wrong number energy groups.".format(nuc.name)
			molar_mass += nuc.a
			for reaction, mgxs in nuc.xs_dict.items():
				for i in range(nuc.g):
					micro_xs[reaction][i] += mgxs[i]
		
		number_density = density*constants.AVOGADRO/molar_mass*1E-24
		macro_xs = {}
		for reaction, mgxs in micro_xs.items():
			macro_xs[reaction] = deepcopy(mgxs_temp)
			for i in range(G):
				macro_xs[reaction][i] = mgxs[i]*number_density
			
		return Material(macro_xs, G, name)
	
	def get_kinf(self):
		if self.groups == 1:
			kinf = float(self.nu_sigma_f/self.sigma_a)
		elif self.groups == 2:
			sigma_s12 = self.scatter_matrix[1, 0] - self.scatter_matrix[0, 1]
			numer = self.nu_sigma_f[0] + (self.nu_sigma_f[1]*sigma_s12)/self.sigma_a[1]
			denom = sigma_s12 + self.sigma_a[0]
			kinf = numer/denom
		else:
			errstr = "Unsure how to do kinf calculation for {} energy gruops."
			raise NotImplementedError(errstr.format(self.groups))
		return kinf
			
		
# test
if __name__ == "__main__":
	# Test materials from nuclides
	n1 = Nuclide(101, {"scatter": [0.0, 1.0]}, g=2)
	n2 = Nuclide(202, {"scatter": [0.2, 1.2], "absorption":[9.0, 0.0]}, g=2)
	nucs = [n1, n2, n2]
	m = Material().fromNuclides(nucs, 1.0)
	print("From nuclides:", m.macro_xs)
	
	# Test materials from pickles
	import pickle
	file2 = open(constants.FNAME2, "rb")
	mg2 = pickle.load(file2)
	file2.close()
	mod = Material(mg2["mod"], name="Moderator (from Pickle)")
	print("From pickle:", mod.macro_xs)

