# Quadrature
#
# Classes and data for quadrature sets

from numpy.polynomial.legendre import leggauss


class Quadrature(object):
	"""Constructor Class"""
	def __init__(self, N):
		assert not N % 2, "SN requires an even number of discrete angles."
		self.N = N
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
		