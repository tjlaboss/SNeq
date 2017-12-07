# Build Model
#
# Construct an OpenMC square pincell. Tally it to get 2-group cross sections.

import openmc
import openmc.mgxs as mgxs
from numpy import array
import pickle

# Run settings
EXPORT = True
STATEPOINT = "statepoint.1000.h5"
# Problem parameters
DIMENSIONS = 1  # use 1D or 2D for purposes of this project
WIDTH = 0.8     # cm; width of one side of the pin cell
PITCH = 1.25    # cm; pin pitch
# Use the CASMO 2-group energy structure
two_groups = mgxs.EnergyGroups()
two_groups.group_edges = array([0.0, 0.625, 20E6])  # eV

# Load materials and export
"""
These materials are from the VERA Core Physics Benchmark Suite.
CASL AMA Benchmark Problem 1A - Pin Cell- Public
  
  tfuel    565 K
  modden   0.743         ! g/cc
  boron    1300          ! ppm
  fuel U31 10.257 94.5 / 3.1 u-234 0.026347
"""
vera_materials = open("vera_materials.dat", "rb")
mats = pickle.load(vera_materials, fix_imports=False)
vera_materials.close()
uo2 = mats["U31"]       # 3.1% enriched uranium dioxide
water = mats["mod"]     # borated water
water._sab = []         # sorry for this, have to work around a pickle bug
water.add_s_alpha_beta("c_H_in_H2O")
openmc_materials = openmc.Materials([uo2, water])

# Build the geometry
cell_universe = openmc.Universe(1)
# Square fuel cell
fuel_w = openmc.XPlane(x0=-WIDTH/2.0, name="Fuel: West edge")
fuel_e = openmc.XPlane(x0=+WIDTH/2.0, name="Fuel: East edge")
fuel_s = openmc.YPlane(y0=-WIDTH/2.0, name="Fuel: South edge")
fuel_n = openmc.YPlane(y0=+WIDTH/2.0, name="Fuel: North edge")
# Will need top and bottom surfaces for 3D.
fuel_region = +fuel_w & -fuel_e & +fuel_s & -fuel_n
fuel_cell = openmc.Cell(name="Fuel pin")
fuel_cell.fill = uo2
fuel_cell.region = fuel_region
cell_universe.add_cell(fuel_cell)

# Square moderator cell
mod_w = openmc.XPlane(x0=-PITCH/2.0, name="Moderator: West edge")
mod_e = openmc.XPlane(x0=+PITCH/2.0, name="Moderator: East edge")
mod_s = openmc.YPlane(y0=-PITCH/2.0, name="Moderator: South edge")
mod_n = openmc.YPlane(y0=+PITCH/2.0, name="Moderator: North edge")
# Again, will need top and bottom for 3D
mod_region = +mod_w & -mod_e & +mod_s & -mod_n
mod_cell = openmc.Cell(name="Surrounding moderator")
mod_cell.fill = water
mod_cell.region = mod_region
cell_universe.add_cell(mod_cell)

# Build the root universe
openmc_geometry = openmc.Geometry()
root_universe = openmc.Universe(0)
root_cell = openmc.Cell(0)
infinite_p = WIDTH/2.0
finite_p = PITCH/2.0
root_w = openmc.XPlane(x0=-finite_p, boundary_type="reflective", name="XMIN")
root_e = openmc.XPlane(x0=+finite_p, boundary_type="reflective", name="XMAX")
if DIMENSIONS <= 1:
	y_p = infinite_p
else:
	y_p = finite_p
root_s = openmc.YPlane(y0=-y_p, boundary_type="reflective", name="YMIN")
root_n = openmc.YPlane(y0=+y_p, boundary_type="reflective", name="YMAX")
if DIMENSIONS <= 2:
	z_p = infinite_p
else:
	z_p = finite_p
root_bot = openmc.ZPlane(z0=-z_p, boundary_type="reflective", name="ZMIN")
root_top = openmc.ZPlane(z0=+z_p, boundary_type="reflective", name="ZMAX")

root_cell.region = +root_w & -root_e & +root_s & -root_n & +root_bot & -root_top
root_cell.fill = cell_universe
root_universe.add_cell(root_cell)
openmc_geometry.root_universe = root_universe

# Do the cross section tallies
pcm_trigger = openmc.Trigger("std_dev", 2E-5)
lib = mgxs.Library(openmc_geometry)
lib.energy_groups = two_groups
lib.mgxs_types = ['total', 'nu-fission', 'transport', 'chi', 'consistent nu-scatter matrix']
lib.by_nuclide = False
lib.domain_type = "material"
lib.domains = [uo2, water]
lib.tally_trigger = pcm_trigger
lib.build_library()
openmc_tallies = openmc.Tallies()
lib.add_to_tallies_file(openmc_tallies)

# Monte Carlo parameters and such
openmc_settings = openmc.Settings()
openmc_settings.batches = 100
openmc_settings.trigger_max_batches = 10*openmc_settings.batches
openmc_settings.inactive = 35
openmc_settings.particles = int(1E7)
openmc_settings.trigger_active = True
openmc_settings.run_mode = "eigenvalue"
openmc_settings.temperature = {"default": uo2.temperature,
                               "method": "interpolation"}
uniform_dist = openmc.stats.Box([-infinite_p]*3, [infinite_p]*3, only_fissionable=True)
source_box = openmc.source.Source(space=uniform_dist)
openmc_settings.source = source_box

if EXPORT:
	openmc_materials.export_to_xml()
	openmc_geometry.export_to_xml()
	openmc_tallies.export_to_xml()
	openmc_settings.export_to_xml()
	lib.dump_to_file("material_lib")

