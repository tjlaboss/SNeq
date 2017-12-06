# Quadrature
#
# Classes and data for quadrature sets

from numpy.polynomial.legendre import leggauss


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
		