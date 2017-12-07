# Load MGXS
#
# Read the OpenMC statepoint and return the two-group cross sections.

import openmc
import openmc.mgxs as mgxs
from openmc_model.group_structure import ngroups
import pickle

STATEPOINT = "statepoint.0100.h5"
FNAME1 = "../mgxs_1group.pkl"
FNAME2 = "../mgxs_2group.pkl"

xsdict_1group = {"mod": dict(),
                 "fuel": dict()}
xsdict_2group = {"mod": dict(),
                 "fuel": dict()}

lib = mgxs.Library.load_from_file(filename="material_lib")
sp = openmc.StatePoint(STATEPOINT)
lib.load_from_statepoint(sp)
for xstype in lib.mgxs_types:
	for domain in lib.domains:
		mg2 = lib.get_mgxs(domain, xstype)
		mg1 = mg2.get_condensed_xs(ngroups[1])
		if domain.name == "U31":
			xsdict_2group["fuel"][mg2.rxn_type] = mg2.get_xs()
			xsdict_1group["fuel"][mg1.rxn_type] = mg1.get_xs()
		else:
			xsdict_2group["mod"][mg2.rxn_type] = mg2.get_xs()
			xsdict_1group["mod"][mg1.rxn_type] = mg1.get_xs()
		
# And export
file1 = open(FNAME1, "wb")
file1.write(pickle.dumps(xsdict_1group))
file1.close()
print("One-group cross sections exported to", FNAME1)
file2 = open(FNAME2, "wb")
file2.write(pickle.dumps(xsdict_2group))
file2.close()
print("One-group cross sections exported to", FNAME2)

