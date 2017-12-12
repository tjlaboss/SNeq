# Problems
#
# Matrix builders for 2-group diffusion problems

#from material import materials
from numpy import array

class Problem(object):
	def __init__(self, dxs, widths, mats, bcs):
		assert len(bcs) == 2, \
			"Exactly 2 boundary conditions must be given (left, right)"
		assert len(dxs) == len(widths) == len(mats), \
			"An unequal number of zone properties (dx, width, material) has been given"
		
		self.nzones = len(dxs)
		self.bcs = bcs
		self.dxs = dxs
		self.widths = widths
		self.mats = mats
		
		self.nxs = self.widths/self.dxs
		self.total_nx = self.nxs.sum()
		self.indicies = array(self.nxs, dtype = int)
		for i in range(len(self.nxs)):
			self.indicies[i] += self.nxs[:i].sum()
		self.total_width = sum(widths)
	
	def index_by_x(self, x):
		"""Given an x position, return the index of the zone it falls in"""
		errstr = "x must be between 0 and " + str(self.total_width) + " (the total width)"
		assert 0 <= x <= self.total_width, errstr
	
	def properties(self, i):
		"""Given the index i, return the properties at i"""
		#n = [j for j,k in enumerate(self.nxs) if k > i][0]
		for j, k in enumerate(self.indicies):
			if k > i:
				return j
		# If this loop terminates without returning a value, indexing is messed up
		errstr = "Problem.properties() was unable to find a node at index " + str(i)
		raise IndexError(errstr)
	
	def distance(self, i):
		"""Given the index i, return the x distance at i"""
		for j, k in enumerate(self.indicies):
			if k > i:
				xdist = sum(self.dxs[:j]*self.nxs[:j])
				xdist += self.dxs[j]*(i - sum(self.nxs[:j]))
				return xdist
		errstr = "Problem.properties() was unable to find a node at index " + str(i)
		raise IndexError(errstr)


	