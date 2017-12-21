# Quadrature
#
# Classes and data for quadrature sets

from numpy.polynomial.legendre import leggauss
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
		
		
class LevelSymmetricQuadrature(Quadrature):
	"""3-D quadrature set using the Level-Symmetric scheme
	
	Values are only available for up to S24.
	
	Indexing for reflection uses the following scheme:
		0 -> npq:       +mux, -muy
		npq -> 2*npq:   +mux, +muy
		2*npq -> 3*npq: -mux, -muy
		3*npq -> 4*npq: -mux, +muy

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
	
	def reflect_angle(self, n):
		raise NotImplementedError("Cannot reflect Level-Symmetric angles yet")
#LevelSymmetricQuadrature(16)