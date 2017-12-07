# Load MGXS
#
# Read the OpenMC statepoint and return the two-group cross sections.

import openmc
import openmc.mgxs as mgxs

STATEPOINT = "statepoint.1000.h5"


lib = mgxs.Library.load_from_file(filename="material_lib")
sp = openmc.StatePoint(STATEPOINT)
lib.load_from_statepoint(sp)
for xstype in lib.mgxs_types:
	for domain in lib.domains:
		mg = lib.get_mgxs(domain, xstype)
		print("{}, {}:".format(domain.name, xstype))
		# TODO: get these data and export them.
		mg.print_xs()

