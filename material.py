# Material
#
# Multi-group material classes

import constants
import numpy as np

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
	nuclides:       list of Nuclides in the Material
	density:        float, g/cm^3: mass density of the material
	name:           str; descriptive name of the material
					[Default: empty string]
	
	Attributes:
	-----------
	g:              int; number of energy groups
	molar_mass:     float; mass of one mole of the material.
	number_density: float; number of atoms/cm^3 * 1E-24
	"""
	def __init__(self, nuclides, density, name=""):
		self.density = density
		self.name = name
		
		self.g = nuclides[0].g
		self._micro_xs = {}
		
		mgxs_temp = np.zeros(self.g)
		for reaction in constants.REACTIONS:
			self._micro_xs[reaction] = mgxs_temp[:]
		molar_mass = 0.0
		for nuc in nuclides:
			assert nuc.g == self.g, \
				"Nuclide {} has the wrong number energy groups.".format(nuc.name)
			molar_mass += nuc.a
			for reaction, mgxs in nuc.xs_dict.items():
				for i in range(nuc.g):
					self._micro_xs[reaction][i] += mgxs[i]
		
		self.number_density = density*constants.AVOGADRO/molar_mass*1E-24
		self.macro_xs = {}
		for reaction, mgxs in self._micro_xs.items():
			self.macro_xs[reaction] = mgxs_temp[:]
			for i in range(self.g):
				self.macro_xs[reaction][i] = mgxs[i]*self.number_density
			
		
# test
if __name__ == "__main__":
	n1 = Nuclide(101, {"scatter": [0.0, 1.0]}, g=2)
	n2 = Nuclide(202, {"scatter": [0.2, 1.2], "absorption":[9.0, 0.0]}, g=2)
	nucs = [n1, n2, n2]
	m = Material(nucs, 1.0)
	print(m.macro_xs)

