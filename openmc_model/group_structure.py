# Group structure
#
# Module with the multigroup structures used in this project

import openmc.mgxs as mgxs
from numpy import array

ngroups = dict()

ngroups[2] = mgxs.EnergyGroups()
ngroups[2].group_edges = array([0.0, 0.625, 20E6])  # eV

ngroups[1] = mgxs.EnergyGroups()
ngroups[1].group_edges = array([0.0, 020E6])  # eV
