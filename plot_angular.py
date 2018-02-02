# Plot Angular
#
# Plot discrete anglar fluxes

from pylab import *

def plot_1group_angular_flux(tally, theta0=0.0, title_text=""):
	figure()
	ax = subplot(111, projection="polar")
	psi = tally.evaluate(True)
	nflux = len(psi)
	
	avals = 2*pi/nflux*array([n for n in linspace(0, nflux, nflux+1)])
	avals += theta0
	pvals = zeros(nflux+1)
	pvals[:nflux] = psi
	# Link the angles back up
	pvals[-1] = pvals[0]
	
	# Make pretty plots
	ax.plot(avals, pvals, "o-")
	ax.set_title(title_text + tally.name + "\n", fontweight="bold", fontsize=14)
	ax.set_rticks(linspace(0, 2, 5))
	tight_layout()
	#show()
	

