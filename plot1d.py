# Plot, 1D
#
# 1-D slab flux plot

from pylab import *

def _setup_plot(problem, xvals, nxmod):
	m = 1.1*problem.flux.max()
	if nxmod:
		plot([nxmod, nxmod], [0, m], "gray")
		plot([problem.nx - nxmod - 1, problem.nx - nxmod - 1], [0, m], "gray")
	xlabel("Node number", fontsize=14)
	ylabel("$\overline{\phi(x)}$", fontsize=14)
	titstr = "$S_{" + str(problem.quad.N) + "}$, slab"
	title(titstr, fontsize=18)
	xlim([xvals[0], xvals[-1]])
	ylim([0, m])
	grid()
	show()

def plot_1group_flux(problem, normalize = False, nxmod = None):
	phi = problem.flux
	if normalize:
		phi /= problem.flux.max()
	xvals = range(0, problem.nx)
	plot(xvals, phi, "b-o")
	_setup_plot(problem, xvals, nxmod)
	

def plot_2group_flux(problem, normalize = False, nxmod = None):
	if normalize:
		problem.flux /= problem.flux.max()
	phi1 = problem.flux[:, 0]
	phi2 = problem.flux[:, 1]
	xvals = range(0, problem.nx)
	plot(xvals, phi1, "b-o", label="Fast flux")
	plot(xvals, phi2, "r-o", label="Thermal flux")
	legend()
	_setup_plot(problem, xvals, nxmod)

