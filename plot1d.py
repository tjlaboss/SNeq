# Plot, 1D
#
# 1-D slab flux plot

from pylab import *

def plot_1group_flux(problem, nxmod = None):
	phi = problem.flux/problem.flux.max()
	xvals = range(0, problem.nx)
	plot(xvals, phi, "b-o")
	m = 1.1  # phi.max()
	if nxmod:
		plot([nxmod, nxmod], [0, m], "gray")
		plot([problem.nx - nxmod - 1, problem.nx - nxmod - 1], [0, m], "gray")
	xlabel("Node number", fontsize=14)
	ylabel("$\overline{\phi(x)}$", fontsize=14)
	titstr = "$S_{" + str(problem.quad.N) + "}$, slab"
	title(titstr, fontsize=18)
	ylim([0, 1.1])
	grid()
	show()
