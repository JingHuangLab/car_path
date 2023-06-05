# Integrated test for Alanine dipeptide isomerization. 
# Zilin Song, 5 Jul 2022
# 

# Generic Python
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
from typing import Tuple, Type

# PyCoSPath
from pycospath.cos import   ConstAdvRep          as CAR,                       \
                            ConstAdvRepParamTan  as carpt,                     \
                            ReaxPathCons         as rpc,                       \
                            ReaxPathConsGradProj as rpcgp,                     \
                            StringMethod         as sm,                        \
                            StringMethodGradProj as smgp,                      \
                            CosBase
from pycospath.utils import rigid_fit
from pycospath.chain import ChainCart
from pycospath.comms.openmm import RepOpenMM, PotOpenMM
from pycospath.opt import ConsGradientDescent

# Misc.
import numpy

# OpenMM helps. 
from openmm import VerletIntegrator, unit, Context as OmmContext, System as OmmSystem
from openmm.unit import Quantity as OmmQuantity
from openmm.app import CharmmPsfFile, PDBFile, CharmmParameterSet, NoCutoff

def mk_chain(n_reps: int = 25) -> ChainCart:
    """Make a list of replicas that linearly intercepts the cartesian space b/w
    the C_7eq and C_ax isomers of alanine dipeptide. """
    ffs = CharmmParameterSet(
                    "./examples/toppar/top_all36_prot.rtf",
                    "./examples/toppar/par_all36m_prot.prm")

    psf = CharmmPsfFile("./examples/omm_c36m_diala/diala.psf")

    mscale_vec = []
    for i_atom in psf.topology.atoms():
        if i_atom.name[0] == 'H':
            mscale_vec.append(1.)
        else:
            mscale_vec.append(1.)
    mscale_vec = numpy.asarray(mscale_vec)

    omm_sys: OmmSystem = psf.createSystem(ffs, nonbondedMethod=NoCutoff)
    omm_cnt = OmmContext(omm_sys, VerletIntegrator(1.*unit.picosecond))
    omm_pot = PotOpenMM(omm_cnt)

    rep_list = []

    pdb_ref = PDBFile(f"./examples/omm_c36m_diala/init_guess/r0.pdb")
    rep_vec_ref = numpy.asarray(pdb_ref.positions.value_in_unit(unit.angstrom)).flatten()

    for i_rep in range(n_reps):
        pdb = PDBFile(f"./examples/omm_c36m_diala/init_guess/r{i_rep}.pdb")
        rep_vec= numpy.asarray(pdb.positions.value_in_unit(unit.angstrom)).flatten()
        rep_coor_align, _, _ = rigid_fit(rep_vec_ref, rep_vec)
        rep_coor = OmmQuantity(rep_coor_align.reshape(-1, 3), unit=unit.angstrom)
        rep_list.append(RepOpenMM(rep_coor, omm_pot, mscale_vec=mscale_vec))

    chain = ChainCart(rep_list)

    return chain

def get_dihe(chain_vec:numpy.ndarray)-> Tuple[float, float]:
    """Compute the psi phi dihedral angles from chain_vec."""

    def compute_dihedral(p0:numpy.ndarray, 
                         p1:numpy.ndarray, 
                         p2:numpy.ndarray, 
                         p3:numpy.ndarray):
        """Compute the dihedral beween four Cartesian points: p0, p1, p2, p3."""
        B0 = p1 - p0
        B1 = p2 - p1
        B2 = p3 - p2

        cosine =   numpy.dot(numpy.cross(B0, B1), numpy.cross(B1, B2))         \
                 / numpy.linalg.norm(numpy.cross(B0, B1))                      \
                 / numpy.linalg.norm(numpy.cross(B1, B2))

        sine   =   numpy.dot(
                    numpy.cross(numpy.cross(B0, B1), numpy.cross(B1, B2)), 
                    B1)                                                        \
                 / numpy.linalg.norm(numpy.cross(B0, B1))                      \
                 / numpy.linalg.norm(numpy.cross(B1, B2))                      \
                 / numpy.linalg.norm(B1)

        dihe = numpy.arctan2(sine, cosine)
        return dihe
    
    # phi: 4, 6,  8, 14, x
    # psi: 6, 8, 14, 16, y
    phis = []
    psis = []

    for rep_vec in chain_vec:

        rep_vec_cart = rep_vec.reshape(-1, 3)

        phi = compute_dihedral(rep_vec_cart[4,  :], rep_vec_cart[6,  :], 
                               rep_vec_cart[8,  :], rep_vec_cart[14, :]
                               ) / numpy.pi * 180.
        
        psi = compute_dihedral(rep_vec_cart[6,  :], rep_vec_cart[8,  :], 
                               rep_vec_cart[14, :], rep_vec_cart[16, :]
                               ) / numpy.pi * 180.
        
        phis.append(phi)
        psis.append(psi)
    
    return phis, psis

def exec_cos(cosname: str):
    """Execute the chain-of-states calculation. """
    # Control vars.

    basedir = "./ms0_car/fig_omm/diala_c36m/intgr_test_results"

    n_steps = 1000

    chain = mk_chain()

    cos = CAR  (chain, verbose=True, cons_log_writer=open(f'{basedir}/{cosname}_itr_cons.log', 'w')) if cosname == 'car'   else \
          carpt(chain, verbose=True, cons_log_writer=open(f'{basedir}/{cosname}_itr_cons.log', 'w')) if cosname == 'carpt' else \
          rpc  (chain, verbose=True, cons_log_writer=open(f'{basedir}/{cosname}_itr_cons.log', 'w')) if cosname == 'rpc'   else \
          rpcgp(chain, verbose=True, cons_log_writer=open(f'{basedir}/{cosname}_itr_cons.log', 'w')) if cosname == 'rpcgp' else \
          sm   (chain, verbose=True                                                                ) if cosname == 'sm'    else \
          smgp (chain, verbose=True)

    opt = ConsGradientDescent(cos)

    phi_list = []
    psi_list = []
    ene_list = []
    rms_list = []

    # print(get_dihe(chain.get_chain_vec()))

    for _ in range(n_steps):
        opt.step()

        phi, psi = get_dihe(chain.get_chain_vec())

        phi_list.append(phi)
        psi_list.append(psi)
        ene_list.append(chain.get_eners_grads()[0])
        rms_list.append(chain.get_rms(chain.get_chain_vec()))

    opt._gd_eta_scal = .95

    for _ in range(100):
        opt.step()
        phi, psi = get_dihe(chain.get_chain_vec())

        phi_list.append(phi)
        psi_list.append(psi)
        ene_list.append(chain.get_eners_grads()[0])
        rms_list.append(chain.get_rms(chain.get_chain_vec()))

    # Save npy. 
    numpy.save(f"{basedir}/{cosname}_phi_traj.npy", numpy.asarray(phi_list))
    numpy.save(f"{basedir}/{cosname}_psi_traj.npy", numpy.asarray(psi_list))
    numpy.save(f"{basedir}/{cosname}_ene_traj.npy", numpy.asarray(ene_list))
    numpy.save(f"{basedir}/{cosname}_rms_traj.npy", numpy.asarray(rms_list))


if __name__ == "__main__":

    # > ./examples/omm_c36m_diala/intgr_test_results/car_itr_traj.log
    exec_cos('car')
    exec_cos('carpt')
    exec_cos('rpc')
    exec_cos('rpcgp')
    exec_cos('sm')
    exec_cos('smgp')
