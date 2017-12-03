# Problem specification
#
# Constants, dimensions, etc. for the solver
# Expect this file to change greatly over the course of its development

RHO_FUEL = 10.4     # g/cm^3
RHO_MOD = 0.7       # g/cm^3

# One-group cross sections
SIGMA_S_U238 = 11.29    # b
SIGMA_S_O16 = 3.888     # b
SIGMA_S_H1 = 20.47      # b
SIGMA_A_H1 = 1.0        # b
# --> fuel is pure scattering; only absorption is hydrogen

# Cell dimensions
PITCH = 1.25            # cm; pin pitch
WIDTH = 0.80            # cm; length of one side of the square fuel pin
