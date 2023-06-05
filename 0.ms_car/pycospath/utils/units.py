################################################################################
#      pyCosPath: A Python Library for Locating Optimal Reaction Pathways      #
#                 in Simulations with Chain-of-States Methods                  #
#                                                                              #
#                     Copyright (c) 2022 the Authors                           #
#                                                                              #
################################################################################
#                                                                              #
# Authors: Zilin Song                                                          #
# Contributors:                                                                #
#                                                                              #
################################################################################

# Implements a series of package specific unit converters. 
# Ratios were obtained from NIST refernce https://www.nist.gov/pml.

import numpy

# Atomic units
BOHRR   = 0.529177210903 # 1 Bohr radius = 0.529177210903 Angstrom
HARTREE = 627.50956      # 1 Hartree     = 627.50956      kcal/mol

def au_dist_to_akma(arr: numpy.ndarray) -> numpy.ndarray:
    """Convert the input distance array in the unit of Bohr to Angstrom."""
    return arr * BOHRR

def akma_dist_to_au(arr: numpy.ndarray) -> numpy.ndarray:
    """Convert the input distance array in the unit of Angstrom to Bohr."""
    return arr / BOHRR

def au_ener_to_akma(arr: numpy.ndarray) -> numpy.ndarray:
    """Convert the input energy array in the unit of Hartree to kcal/mol."""
    return arr * HARTREE

def au_grad_to_akma(arr: numpy.ndarray) -> numpy.ndarray:
    """Convert the input gradient array in Hartree/Bohr to kcal/mol/Angstrom."""
    return arr * HARTREE / BOHRR
