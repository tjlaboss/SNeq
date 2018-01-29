# Quadrature
#
# Classes and data for quadrature sets

from numpy.polynomial.legendre import leggauss
from constants import EDGES
import _level_symmetric


class Quadrature(object):
	"""Constructor Class"""
	def __init__(self, N):
		assert not N % 2, "SN requires an even number of discrete angles."
		self.N = N
		self.N2 = N // 2
		pass


class GaussLegendreQuadrature(Quadrature):
	"""1-D quadrature set using Gauss-Legendre polynomials
	
	Parameter:
	----------
	N:          int (multiple of 2); number of discrete angles
	
	"""
	def __init__(self, N):
		super().__init__(N)
		self.mus, self.weights = leggauss(N)
		self.Nflux = self.N
	
	def __str__(self):
		rep = "1-D Gauss-Legendre Quadrature\n"
		rep += "------------------------------"
		rep += "\nS{}".format(self.N)
		rep += "\ncosines: {}".format(self.mus)
		rep += "\nweights: {}".format(self.weights)
		return rep
	
	def reflect_angle(self, n):
		"""Given an index 'n' from this quadrature, find
		the corresponding cosine in the reflected direction.
		
		Parameter:
		----------
		n:          int; incoming angular flux index
		
		Returns:
		--------
		m:          int; outgoing reflected flux index
		"""
		return (self.N - 1) - n
		
		
class LevelSymmetricQuadrature2D(Quadrature):
	"""2-D quadrature set using the Level-Symmetric scheme
	
	Values are only available for up to S24.
	
	Indexing for reflection uses the following scheme:
		0 -> npq:       +mux, -muy
		npq -> 2*npq:   +mux, +muy
		2*npq -> 3*npq: -mux, +muy
		3*npq -> 4*npq: -mux, -muy

	Parameter:
	----------
	N:          int (multiple of 2); number of discrete angles
	
	Attributes:
	N
	npq;        int; number of flux angles per quadrant
	Nflux:      int; total number of flux angles
	"""
	def __init__(self, N):
		super().__init__(N)
		assert N <= 24, \
			"Level-Symmetric Quadrature is only available for S2 through S24."
		# Populate according to the LS set
		sdict = _level_symmetric.LevelSymmetricQuadrature().getQuadratureSet(N)
		self.muxs = sdict["mu"]
		self.muys = sdict["eta"]
		self.muzs = sdict["xi"]
		self.weights = sdict["weight"]
		self.npq = len(self.weights)  # number per quadrant
		self.Nflux = 4*self.npq
	
	def reflect_angle(self, n, side):
		"""Get the angular index corresponding to the outgoing flux
		at a reflective boundary condition.
		
		Parameters:
		-----------
		n:          int; incoming angular flux index
		side:       str; one of "south", "north", "east", "west"
		
		Returns:
		--------
		m:          int; outgoing angular flux index
		"""
		assert side in EDGES, \
			"{} is not a valid side. Must be in: {}".format(side, EDGES)
		index = n % self.npq
		quad_in = n // self.npq
		if side == "west":
			# LHS edge: reflect (-mux -> +mux)
			assert quad_in in (2, 3)
			if quad_in == 2:
				m = n - self.npq
			else:
				m = n + self.npq
		elif side == "east":
			# RHS edge: (+mux -> -mux)
			assert quad_in in (0, 1)
			if quad_in == 1:
				m = n + self.npq
			else:
				m = n - self.npq
		elif side == "north":
			# (+mux -> -muy)
			assert quad_in in (1, 2)
			if quad_in == 2:
				m = n + self.npq
			else:
				m = n - self.npq
		elif side == "south":
			# (-mux -> +muy)
			assert quad_in in (0, 3)
			if quad_in >= 2:
				m = n - self.npq
			else:
				m = n + self.npq
		else:
			errstr = "{} edge is not available in 2D."
			raise NotImplementedError(errstr.format(side))
		return m % self.Nflux
	
	def inverse_reflect_angle(self, m, side):
		"""The inverse of self.reflect_angle()
		
		Parameters:
		-----------
		m:          int; outgoing angular flux index
		side:       str; one of "south", "north", "east", "west"
		
		Returns:
		--------
		n:          int; incoming angular flux index
		"""
		assert side in EDGES, \
			"{} is not a valid side. Must be in: {}".format(side, EDGES)
		quad_out = m // self.npq
		if side == "west":
			assert quad_out in (0, 1)
			if quad_out == 0:
				n = m - self.npq
			else:
				n = m + self.npq
		elif side == "east":
			assert quad_out in (2, 3)
			if quad_out == 2:
				n = m - self.npq
			else:
				n = m + self.npq
		elif side == "north":
			assert quad_out in (0, 3)
			if quad_out == 0:
				n = m + self.npq
			else:
				n = m - self.npq
		elif side == "south":
			assert quad_out in (1, 2)
			if quad_out == 1:
				n = m - self.npq
			else:
				n = m + self.npq
		else:
			errstr = "{} edge is not available in 2D."
			raise NotImplementedError(errstr.format(side))
		return n % self.Nflux
		
	

# Test reflection
if __name__ == "__main__":
	g = LevelSymmetricQuadrature2D(4)
	for side in EDGES:
		for i in range(g.Nflux):
			s = "{} side:\tAngle: {}\treflected -> {}"
			try:
				j = g.reflect_angle(i, side)
				k = g.inverse_reflect_angle(j, side)
				if k != i:
					print("Reflection error! Predicted incoming =", k)
			except AssertionError:
				pass
			else:
				print(s.format(side, i, j))
		print()
