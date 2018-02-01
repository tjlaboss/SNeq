# Plot Angular
#
# Plot discrete anglar fluxes

from pylab import *

def plot_1group_angular_flux(problem, i, j, location=""):
	figure()
	ax = subplot(111, projection="polar")
	nflux = problem.quad.Nflux
	avals = 2*pi/nflux*array([n for n in linspace(0, nflux, nflux+1)])
	avals += arctan(problem.quad.muys[0]/problem.quad.muxs[0])
	pvals = zeros(nflux+1)
	psi = problem.psi[i, j, :, 0]
	c = 0
	for q in range(4):
		n0 = q*problem.quad.npq
		for a in range(problem.quad.npq):
			if problem.quad.muzs[a] == problem.quad.muzs[a].min():
				pvals[c] = psi[n0 + a]
				c += 1
	# Link the angles back up
	pvals[-1] = pvals[0]
	
	# Make pretty plots
	ax.plot(avals, pvals, "o-")
	title_text = "$\\bf S_{}$ Angular Flux: ".format(problem.quad.N)
	ax.set_title(title_text + location + "\n", fontweight="bold", fontsize=14)
	ax.set_rticks(linspace(0, 2, 5))
	tight_layout()
	#show()
	

