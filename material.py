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
		self.macro_xs = macro_xs
		self.groups = groups
		self.name = name
	
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

