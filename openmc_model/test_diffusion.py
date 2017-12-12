

from diffusion import *
from collections import OrderedDict
import matplotlib.pyplot as plt
from material import Material

data = array([
	# D1    D2     A1      A2     S12      nF1     nF2
	[1.43, 0.37, 0.0079, 0.0605, 0.0195, 0.0034, 0.0750],
	[1.43, 0.37, 0.0084, 0.0741, 0.0185, 0.0054, 0.1010],
	[1.43, 0.37, 0.0089, 0.0862, 0.0178, 0.0054, 0.1010],
	[1.43, 0.37, 0.0088, 0.0852, 0.0188, 0.0062, 0.1249],
	[1.26, 0.27, 0.0025, 0.0200, 0.0294, 0, 0],
	[1.00, 0.34, 0.0054, 0.1500, 0.0009, 0, 0],
	[1.55, 0.27, 0.0010, 0.0300, 0.0500, 0, 0]
])

# Instantiate blank materials
fuel_016 = Material(name="Fuel - 1.6% enriched", groups=2)
fuel_024 = Material(name="Fuel - 2.4% enriched", groups=2)
fuel_bp = Material(name="Fuel - 2.4% with BP", groups=2)
fuel_031 = Material(name="Fuel - 3.1% enriched", groups=2)
baffle_refl = Material(name="Baffle/Reflector", groups=2)
baffle = Material(name="Baffle", groups=2)
refl = Material(name="Reflector", groups=2)
mattuple = (fuel_016, fuel_024, fuel_bp, fuel_031, baffle_refl, baffle, refl)

# Populate the dictionary of materials
materials = OrderedDict()
for k in range(len(mattuple)):
	m = mattuple[k]
	mid = k + 1
	m.key = mid
	
	# Diffusion Coefficient
	m.D = array([data[k, 0],
	             data[k, 1]])
	
	# Transport cross section
	m.sigma_tr = 1.0/(3*m.D)
	
	# Absorption
	m.sigma_a = array([data[k, 2],
	                   data[k, 3]])
	
	# Scattering matrix
	s12 = data[k, 4]
	s11, s22 = m.sigma_tr - m.sigma_a
	s11 -= s12
	m.scatter_matrix = array([
		[s11, 0.0],
		[s12, s22]])
	# And downscatter cross section
	m.sigma_s12 = array([s12, 0.0])
	
	# Nu-Fission
	m.nu_sigma_f = array([data[k, 5],
	                      data[k, 6]])
	
	materials[mid] = m

# Define each of the problems to solve

biglist = array([materials[7], materials[6], materials[4], materials[3], materials[4],
                 materials[6], materials[7]])

prob0 = Problem(bcs=("reflective", "reflective"), dxs=array([60]), widths=array([300]), mats=array([materials[4]]))

prob1 = Problem(bcs=("vacuum", "vacuum"), dxs=array([5]), widths=array([300]), mats=array([materials[1]]))

prob2 = Problem(bcs=("vacuum", "vacuum"), dxs=array([2.5, 5, 2.5]),
                widths=array([25, 250, 25]),
                mats=array([materials[2]]*3))

prob3 = Problem(bcs=("vacuum", "vacuum"), dxs=array([2.5, 5, 2.5]),
                widths=array([25, 250, 25]),
                mats=array([materials[5], materials[2], materials[5]]))

prob4 = Problem(bcs=("vacuum", "vacuum"), dxs=array([1., 1., 1.]),
                widths=array([25, 250, 25]),
                mats=array([materials[5], materials[2], materials[5]]))

prob5 = Problem(bcs=("vacuum", "vacuum"), dxs=array([1.]*5),
                widths=array([25, 16, 218, 16, 25]),
                mats=array([materials[5], materials[4], materials[3], materials[4], materials[5]]))

prob6 = Problem(bcs=("vacuum", "vacuum"), dxs=array([1.]*7),
                widths=array([23, 2, 14, 222, 14, 2, 23]),
                mats=biglist)

prob7 = Problem(bcs=("vacuum", "vacuum"), dxs=array([1.]*7),
                widths=array([23, 2, 16, 218, 16, 2, 23]),
                mats=biglist)

prob8 = Problem(bcs=("vacuum", "vacuum"), dxs=array([1.]*7),
                widths=array([23, 2, 18, 214, 18, 2, 23]),
                mats=biglist)

prob9 = Problem(bcs=("reflective", "reflective"),
                dxs=array([1.]*7),
                widths=array([23, 2, 18, 214, 18, 2, 23]),
                mats=biglist)

all_problems = (prob0, prob1, prob2, prob3, prob4, prob5, prob6, prob7, prob8, prob9)

PLOT = True


for i, prob in enumerate(all_problems):
#for i, prob in enumerate([prob9]):
	print("\n\nProblem", i, "\n")
	for m in prob.mats:
		print(m, "k_inf =", m.get_kinf())
	if PLOT:
		plt.figure(i)
	solve_problem(prob, PLOT, i)
if PLOT:
	plt.show()
