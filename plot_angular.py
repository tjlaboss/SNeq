# Plot Angular
#
# Plot discrete anglar fluxes

from pylab import *

def plot_1group_angular_flux(tally, quad, theta0=0, title_text=""):
	"""Plot a 2-D, 1-group angular flux tally
	
	Parameters:
	-----------
	tally:          tallies.AngularFluxTally2D
	quad:           quadratures.Quadrature2D
					(only tested with LevelSymmetricQuadrature2D)
	theta0:         float, radians, optional; offset for the plot.
					[Default: 0]
	title_text:     str, optional; title text to display on the plot
	"""
	figure()
	ax = subplot(111, projection="polar")
	psi = tally.evaluate(True)
	nflux = len(psi)
	
	avals = zeros(nflux)
	c = 0
	for ang in range(quad.Nflux):
		if ang in tally.n:
			theta = theta0 + (ang//quad.npq)*pi/2
			a = ang % quad.npq
			mu = quad.muxs[a]
			eta = quad.muys[a]
			theta += arctan(eta/mu)
			avals[c] = theta
			c += 1
			
	# Sort the flux magnitudes according to their actual angular position
	pvals = [p for a,p in sorted(zip(avals, psi))]
	avals = list(sorted(avals))
	# Link the angles back up
	pvals = array(pvals + [pvals[0]])
	avals = array(avals + [avals[0]])
	
	# Make pretty plots
	ax.plot(avals, pvals, "o-")
	ax.set_title(title_text + tally.name + "\n", fontweight="bold", fontsize=14)
	ax.set_rticks(linspace(0, 2, 5))
	tight_layout()

