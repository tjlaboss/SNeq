# Tallies
#
# Do flux tallies on an SN mesh

import numpy

class Tally(object):
	"""Base class for all tallies"""
	def __init__(self, g, name=""):
		self.g = g
		self.name = name
	
	def _get_energy_index(self, g):
		for i, j in enumerate(self.g):
			if g == j:
				return i
		raise IndexError("Out of bounds: Group {} of {}".format(g, self.g))

class AngularFluxTally(Tally):
	"""Base class for angular flux tallies"""
	def __init__(self, n, g, name = ""):
		super().__init__(g, name)
		self.n = n
	
	def _get_angular_index(self, n):
		for i, j in enumerate(self.n):
			if n == j:
				return i
		raise IndexError("Out of bounds: Angle {} of {}".format(n, self.n))

class AngularFluxTally2D(AngularFluxTally):
	"""Two-dimensional angular flux tally
	
	I plan to have this accept ranges for `i` and `j` as well, but I have
	no need to implement this now. When I do, it may be smarter just
	to code up a mesh tally.
	
	Parameters:
	-----------
	i:              range of ints; x-index to tally at
	j:              range of ints; y-index to tally at
	n:              range of ints; angular indices to tally over
	g:              range of ints; energy indices
	
	"""
	def __init__(self, i, j, n, g, name = ""):
		super().__init__(n, g, name)
		self.i = i
		self.j = j
		self._values = numpy.zeros((len(n), len(g)))
	
	def applies(self, i, j, n, g):
		return i == self.i and j == self.j and n in self.n and g in self.g
			
	def update(self, psi, n, g):
		ndex = self._get_angular_index(n)
		gdex = self._get_energy_index(g)
		self._values[ndex, gdex] = psi
		
	def evaluate(self, merge_energies=False):
		"""Once the simulation is done, return the volume-averaged values
		
		Parameter:
		----------
		merge_energies:     Boolean, optional; whether to merge the energy
							groups in the final reported tally.
							[Default: False]
		
		Returns:
		--------
		if merge_energies:
			1D array; total angular flux over this tally
		else:
			2D array; angular fluxes per energy group over this tally
		"""
		# this is where averaging will take place over i and j ranges
		# without doing that, this function simply returns the angles...
		if merge_energies:
			return self._values.sum(axis=1)
		else:
			return self._values
		
		
		
