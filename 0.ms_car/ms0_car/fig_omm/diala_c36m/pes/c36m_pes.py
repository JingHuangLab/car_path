# Soft PES scan of di-ALA phi-psi dihedrals.
# Ye Ding, 15 Jun 2022.

from openmm import System, State, VerletIntegrator, unit, CustomTorsionForce
from openmm.app import CharmmPsfFile, CharmmCrdFile, CharmmParameterSet, PDBFile
from openmm.app import NoCutoff, Simulation
import numpy, math

# ff, top, and cor.
ffs = CharmmParameterSet("./examples/toppar/top_all36_prot.rtf",
                         "./examples/toppar/par_all36m_prot.prm")
psf = CharmmPsfFile("./examples/omm_c36m_diala/diala.psf")
cor = CharmmCrdFile("./examples/omm_c36m_diala/diala.cor")

# Get atom indices of phi/X, psi/Y dihedrals.
## Phi atom indices with order.
atoms = psf.topology.atoms()
phi_atom_names = ["CLP", "NL", "CA", "CRP"]
phi_atom_indices = []

for name in phi_atom_names:

    for atom in atoms:

        if atom.name == name:
            phi_atom_indices.append(atom.index)
            break

## Psi atom indices with order.
atoms = psf.topology.atoms()
psi_atom_names = ["NL", "CA", "CRP", "NR"]
psi_atom_indices = []

for name in psi_atom_names:

    for atom in atoms:

        if atom.name == name:
            psi_atom_indices.append(atom.index)
            break

# Define harmonic biasing potentials on two dihedrals of dialanine (psi, phi) in the OpenMM system
## for phi/X dihedral
bias_torsion_phi = CustomTorsionForce("0.5*k_phi*dtheta^2; dtheta = min(tmp, 2*pi-tmp); tmp = abs(theta - phi)")
bias_torsion_phi.addGlobalParameter("pi", math.pi)
bias_torsion_phi.addGlobalParameter("k_phi", 100.0)
bias_torsion_phi.addGlobalParameter("phi", 0.0)
## 6, 8, 14, 16 are the indices of the atoms of the torsion phi
bias_torsion_phi.addTorsion(phi_atom_indices[0], phi_atom_indices[1], phi_atom_indices[2], phi_atom_indices[3])

## for psi/X dihedral
bias_torsion_psi = CustomTorsionForce("0.5*k_psi*dtheta^2; dtheta = min(tmp, 2*pi-tmp); tmp = abs(theta - psi)")
bias_torsion_psi.addGlobalParameter("pi", math.pi)
bias_torsion_psi.addGlobalParameter("k_psi", 100.0)
bias_torsion_psi.addGlobalParameter("psi", 0.0)
## 4, 6, 8, 14 are the indices of the atoms of the torsion psi
bias_torsion_psi.addTorsion(psi_atom_indices[0], psi_atom_indices[1], psi_atom_indices[2], psi_atom_indices[3])

# Create OpenMM system with biasing forces.
omm_sys: System = psf.createSystem(ffs, nonbondedMethod=NoCutoff)
omm_sys.addForce(bias_torsion_psi)
omm_sys.addForce(bias_torsion_phi)

# Create OpenMM simulation.
sim = Simulation(psf.topology, omm_sys, VerletIntegrator(1.*unit.picosecond))
sim.context.setPositions(cor.positions)

# PES control vars.
grid_interval = 3. # 15.
grid_points   = int(360./grid_interval)+1
phi_x = numpy.arange(0, grid_points)
psi_y = numpy.arange(0, grid_points)

# Init data mat to be filled at 999. to detect abnormality.
eners    = numpy.ones((grid_points, grid_points)) * 999
phi_degs = numpy.ones((grid_points, )) * 999
psi_degs = numpy.ones((grid_points, )) * 999

for x in phi_x:

    for y in psi_y:
        
        # Compute CV values. 
        phi_degs[x] = x * grid_interval - 180
        psi_degs[y] = y * grid_interval - 180
        phi_rad = phi_degs[x] / 180. * math.pi
        psi_rad = psi_degs[y] / 180. * math.pi

        # Set CV and minimize.
        sim.context.setParameter("phi", phi_rad)
        sim.context.setParameter("psi", psi_rad)
        sim.context.setParameter("k_phi", 500000.0)
        sim.context.setParameter("k_psi", 500000.0)

        sim.minimizeEnergy(tolerance=1e-6*unit.kilocalories_per_mole/unit.angstrom)

        # Remove bias and compute energy.
        sim.context.setParameter("k_phi", 0.)
        sim.context.setParameter("k_psi", 0.)
        state: State = sim.context.getState(getPositions=True, getEnergy=True)
        eners[y][x] = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    
        print(f"phi: {phi_degs[x]}, psi: {psi_degs[y]}, ener: {eners[y][x]}")

        # if abs(phi_degs[x] - 75) <= 1e-2 and abs(psi_degs[y] + 45) <= 1e-2:
        #     print('Writing PDB: C_ax')
        #     PDBFile.writeFile(psf.topology, state.getPositions(), open("./examples/omm_c36m_diala/omm_pes/c36m_c_ax.pdb", 'w'))

        if abs(phi_degs[x] + 80) <= 1e-2 and abs(psi_degs[y] - 80) <= 1e-2:
            print('Writing PDB: C_7eq')
            PDBFile.writeFile(psf.topology, state.getPositions(), open("./examples/omm_c36m_diala/c36m_c_7eq.pdb", 'w'))

# Save for later reuse with numpy.load().
numpy.save("./ms0_car/fig_omm/diala_c36m/pes/c36m_phi.npy", phi_degs)
numpy.save("./ms0_car/fig_omm/diala_c36m/pes/c36m_psi.npy", psi_degs)
numpy.save("./ms0_car/fig_omm/diala_c36m/pes/c36m_ene.npy", eners)
