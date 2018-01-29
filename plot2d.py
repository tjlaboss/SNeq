

from pylab import *

def _setup_plot(problem, xvals, nxmod, show_grid):
	xlabel("x-nodes", fontsize=14)
	ylabel("y-nodes", fontsize=14)
	titstr = "$S_{" + str(problem.quad.N) + "}$, slab"
	title(titstr, fontsize=18)
	xticks(-0.5 + arange(problem.nx+1), range(problem.nx+1))
	yticks(-0.5 + arange(problem.ny+1), range(problem.ny+1))
	if show_grid:
		grid()
	show()

def plot_1group_flux(problem, normalize=False, nxmod = None, grid=True):
	phi = array(problem.flux[:,:,0].T)
	if normalize:
		phi /= phi.mean()
	hotplot = imshow(phi.squeeze(), interpolation = 'none', cmap = 'jet')
	cmax = max(abs(nanmax(phi)), abs(nanmin(phi)))
	cmin = 2 - cmax
	clim(cmax, cmin)
	#title(case_name + " Fission Rates")
	colorbar(hotplot)
	_setup_plot(problem, None, nxmod, grid)