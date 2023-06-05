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

# Imports
from typing import Tuple

from pycospath.comms import PotBase

from openmm import Context  as OmmContext,                                     \
                   System   as OmmSystem,                                      \
                   State    as OmmState
from openmm.unit import Quantity as OmmQuantity

class PotOpenMM(PotBase):
    """The OpenMM potential."""

    def __init__(self, 
                 omm_context: OmmContext):
        """Create an OpenMM potential function.

        Parameters
        ----------
        omm_context: openmm.Context
            It can be created as:
            ```
            toppar = CharmmParameterSet(
                        "./examples/toppar/top_all36_prot.rtf",
                        "./examples/toppar/par_all36m_prot.prm",
            )
            psf = CharmmPsfFile("./examples/omm_c36m_diala/diala.psf")

            omm_integrator = LangevinIntegrator(0*unit.kelvin, 
                                                1./unit.picosecond, 
                                                1.*unit.picosecond)
            omm_system = psf.createSystem(toppar, 
                                          nonbondedMethod=NoCutOff)
            omm_context = Context(omm_sys, integrator)
            ```
        """
        if not isinstance(omm_context, OmmContext):
            raise TypeError("omm_context must be openmm.Context type.")
        
        self._omm_context = omm_context

    def get_omm_context(self) -> OmmContext:
        """Get the OpenMM Context object (self._omm_context).

        Returns
        -------
        omm_context: openmm.Context
            The OpenMM Context object.
        """
        return self._omm_context

    def get_omm_system(self) -> OmmSystem:
        """Get the OpenMM System object (self._omm_context.getSystem()).
        
        Returns
        -------
        omm_system: openmm.System
            The OpenMM System object.
        """
        return self._omm_context.getSystem()

    def get_ener(self, 
                 coor: OmmQuantity
                 ) -> OmmQuantity:
        """Get the energy of the coor openmm.unit.Quantity.
        
        Parameters
        ----------
        coor: openmm.unit.Quantity
            The input system coordinate openmm.unit.Quantity.

        Returns
        -------
        ener: openmm.unit.Quantity
            The energy openmm.unit.Quantity.
        """
        self._omm_context.setPositions(coor)

        omm_state: OmmState = self._omm_context.getState(getEnergy=True)

        ener: OmmQuantity = omm_state.getPotentialEnergy()

        return ener

    def get_ener_grad(self, 
                      coor: OmmQuantity
                      ) -> Tuple[OmmQuantity, OmmQuantity]:
        """Get the energy and gradients of the coor openmm.unit.Quantity.
        
        Parameters
        ----------
        coor: openmm.unit.Quantity
            The input system coordinate openmm.unit.Quantity.

        Returns
        -------
        (ener, grad): (openmm.unit.Quantity, openmm.unit.Quantity)
            ener: The energy openmm.unit.Quantity;
            grad: The gradient openmm.unit.Quantity.
        """
        self._omm_context.setPositions(coor)
        
        omm_state: OmmState = self._omm_context.getState(getEnergy=True, 
                                                         getForces=True)

        ener: OmmQuantity = omm_state.getPotentialEnergy()

        grad: OmmQuantity = -1. * omm_state.getForces(asNumpy=True)    # grad = -force.

        return ener, grad
        