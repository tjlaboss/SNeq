# Problem specification
#
# Constants, dimensions, etc. for the solver
# Expect this file to change greatly over the course of its development

from pincell import Pincell1D
from cmr import RebalancePincell1D
from cmfd import FiniteDifferencePincell1D
import accelerator
import quadrature
import material
import calculator
import plot1d
import constants
import pickle
import numpy as np

FIXED_SOURCE = 0.0
G = 2
# Cell dimensions
PITCH = 20   # cm; pin pitch
WIDTH = 20   # cm; length of one side of the square fuel pin
NXMOD = 0
NXFUEL = 10
BOUNDARIES = ("reflective", "reflective")
#BOUNDARIES = ("vacuum", "vacuum")


# Load the cross sections from disk
if G == 1:
	file1 = open(constants.FNAME1, "rb")
	mg = pickle.load(file1)
	file1.close()
elif G == 2:
	# file2 = open(constants.FNAME2, "rb")
	# mg = pickle.load(file2)
	# file2.close()

	# 3.1% enriched fuel from 22.211/PSet06
	mg = {"fuel": {}, "mod": {}}
	fuel_diffusion = np.array([1.43, 0.37])
	mg["fuel"]["D"] = fuel_diffusion
	mg["fuel"]["transport"] = 1/(3*fuel_diffusion)
	mg["fuel"]["absorption"] = np.array([0.0088, 0.0852]) # 3%
	mg["fuel"]["nu-fission"] = np.array([0.0062, 0.1249])
	
	s12 = 0.0188
	scatter = mg["fuel"]["transport"] - mg["fuel"]["absorption"]
	s11 = scatter[0] - s12
	s22 = scatter[1]
	fuel_scatter_matrix = np.array([[s11,    0],
	                                [s12,  s22]])
	mg["fuel"]["nu-scatter"] = fuel_scatter_matrix
	
	# Water reflector
	mod_diffusion = np.array([1.55, 0.27])
	mg["mod"]["D"] = mod_diffusion
	mg["mod"]["transport"] = 1/(3*mod_diffusion)
	mg["mod"]["absorption"] = np.array([0.0010, 0.0300])
	
	s11, s22 = mg["mod"]["transport"] - mg["mod"]["absorption"]
	s12 = 0.0500
	mod_scatter_matrix = np.array([[s11, 0],
	                               [s12, s22]])
	mg["mod"]["nu-scatter"] = mod_scatter_matrix
	scatter = mod_scatter_matrix.sum(axis=0)
	mg["mod"]["total"] = mg["mod"]["transport"]
else:
	raise NotImplementedError("{} groups".format(G))

mod_mat = material.Material(name="Moderator", macro_xs=mg["mod"], groups=G)
fuel_mat = material.Material(name="Fuel, 3.1%", macro_xs=mg["fuel"], groups=G)

if G == 1:
	# debug
	fuel_mat.nu_sigma_f = 1.1*fuel_mat.sigma_a # debug; force kinf to 2.1
	#fuel_mat.sigma_tr = mg["fuel"]["transport"]
	fuel_mat.scatter_matrix = fuel_mat.sigma_tr - fuel_mat.sigma_a
	# analytically calculate kinf from the 1-group xs


kinf = fuel_mat.get_kinf()
print("kinf = {:1.5f}".format(kinf))


# test fine mesh
s2 = quadrature.GaussLegendreQuadrature(2)
KGUESS = kinf
#KGUESS = 1.0
cell = Pincell1D(s2, mod_mat, fuel_mat, PITCH, WIDTH, NXMOD, NXFUEL, groups=G)
cell.set_bcs(BOUNDARIES)
coarse_mesh = None

# test CMR
coarse_mesh = RebalancePincell1D().fromFineMesh(cell, 2)
cmr = accelerator.RebalanceAccelerator1D(coarse_mesh, cell)

# test CMFD
coarse_mesh = FiniteDifferencePincell1D().fromFineMesh(cell, 2)
cmfd = accelerator.FiniteDifference1D(coarse_mesh, cell)

#solver = calculator.DiamondDifferenceCalculator1D(s2, cell, accelerator=cmr, kguess=KGUESS)
solver = calculator.DiamondDifferenceCalculator1D(s2, cell, accelerator=cmfd, kguess=KGUESS)
solver.transport_sweep(KGUESS)

converged = solver.solve(eps=1E-6, maxiter=1000)
phi = solver.mesh.flux
print(cell)
print(phi)
print(solver.k)
if converged:
	if G == 1:
		plot1d.plot_1group_flux(cell, True, NXMOD)
	elif G == 2:
		plot1d.plot_2group_flux(cell, True, NXMOD)