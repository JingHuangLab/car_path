# INtegrated test for C12 peptide helix unwinding via a Greedy Snake strategy. 
# Zilin Song, 7 Sep 2022
# 

# Generic python.
import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
from typing import List

# PyCoSPath
from pycospath.cos import ConstAdvRep as CAR
from pycospath.utils import rigid_fit
from pycospath.chain import ChainCart
from pycospath.comms.openmm import RepOpenMM, PotOpenMM
from pycospath.opt import ConsGradientDescent, ConsAdaptiveMomentum

# Misc.
import numpy 

# OpenMM helps.
from openmm import VerletIntegrator, unit, Context as OmmContext, System as OmmSystem
from openmm.unit import Quantity as OmmQuantity
from openmm.app import CharmmPsfFile, PDBFile, DCDFile, CharmmParameterSet, NoCutoff, GBn

# OpenMM potential function (shared by all replicas).
ffs = CharmmParameterSet("./examples/toppar/top_all36_prot.rtf",
                         "./examples/toppar/par_all36m_prot.prm")
psf = CharmmPsfFile("./examples/omm_c36m_c12pep/c12pep.psf")
omm_sys: OmmSystem = psf.createSystem(ffs, nonbondedMethod=NoCutoff, implicitSolvent=GBn)
omm_cnt = OmmContext(omm_sys, VerletIntegrator(1.*unit.picosecond))
omm_pot = PotOpenMM(omm_cnt)

# Make mass-scaling functions -> H does not contribute to the RMS and is free of constraints. 
mscale_vec = []
for i_atom in psf.topology.atoms():
    if i_atom.name[0] == 'H':
        mscale_vec.append(0.)
    else:
        mscale_vec.append(1.)
mscale_vec = numpy.asarray(mscale_vec)

# Some control variables.
thresh_consume_rms = .02 # The consumption chain will have rms values lower than
                         # this threshold after the linear interception. 

log_out = open('./ms0_car/fig_omm/c12pep_c36m/greedy_snake_refine.log', 'w')

def mk_rep(rep_vec: numpy.ndarray) -> RepOpenMM:
    """Make a RepOpenMM object with the given rep_vec.
    rep_vec is first transformed to an OmmQuantity object which is used to init
    the RepOpenMM object.
    """
    rep_coor = OmmQuantity(rep_vec.reshape(-1, 3), unit=unit.angstrom)
    rep = RepOpenMM(rep_coor, omm_pot, mscale_vec = mscale_vec) # H atoms does not contribute to RMS. 

    return rep

def exec_cos(save_tag:     int,
             rep_list:     List[RepOpenMM], 
             ) -> List[RepOpenMM]: 
    """Execute chain-of-states optimizations with kwarg specs."""
    chain = ChainCart(rep_list)
    cos = CAR(chain, verbose=False, cons_curvreglr_thresh=15)

    opt_adam = ConsAdaptiveMomentum(cos, fix_mode = 'none')

    lowest_max_ener = 0
    lowest_max_ener_cvec = -1

    for _ in range(1000):
        
        if _ % 1 == 0:
            print(f'  AdaM step: {_-1} / 1000. Max ENER: {numpy.max(opt_adam._y_pv)}')
            log_out.write(f'  AdaM step: {_-1} / 1000. Max ENER: {numpy.max(opt_adam._y_pv)}\n')
            log_out.flush()
        
        opt_adam.step()

        if numpy.max(opt_adam._y_pv) < lowest_max_ener:
            lowest_max_ener = numpy.max(opt_adam._y_pv)
            lowest_max_ener_cvec = opt_adam.get_cos().get_chain().get_chain_vec()

    cos.get_chain().set_chain_vec(lowest_max_ener_cvec)

    opt_grad = ConsGradientDescent(cos, gd_eta=.001, gd_eta_scal=0.98, fix_mode = 'both')

    lowest_max_ener = 0
    lowest_max_ener_cvec = -1

    for _ in range(200):
        
        if _ % 10 == 1:
            print(f'  GraD step: {_-1} / 200. Max ENER: {numpy.max(opt_grad._y_pv)}')
            log_out.write(f'  GraD step: {_-1} / 200. Max ENER: {numpy.max(opt_grad._y_pv)}\n')
            log_out.flush()

        opt_grad.step()

        if numpy.max(opt_grad._y_pv) < lowest_max_ener:
            lowest_max_ener = numpy.max(opt_grad._y_pv)
            lowest_max_ener_cvec = opt_grad.get_cos().get_chain().get_chain_vec()

    chain_vec = chain.get_chain_vec()
    abslen = numpy.linalg.norm(chain_vec[-1] - chain_vec[0])
    rms    = chain.get_rms(chain_vec)
    ener   = chain.get_eners_grads()[0]

    if save_tag != 'nosave':
        numpy.save(f'./ms0_car/fig_omm/c12pep_c36m/snake_path/snake_path_{save_tag}.npy', chain_vec)
        numpy.save(f'./ms0_car/fig_omm/c12pep_c36m/snake_path/snake_path_{save_tag}_min.npy', lowest_max_ener_cvec)

        dcd = DCDFile(open(f'./ms0_car/fig_omm/c12pep_c36m/snake_path/snake_path_{save_tag}.dcd', 'wb'), psf.topology, .1)
        
        for i_rvec in range(chain_vec.shape[0]):
            dcd.writeModel(OmmQuantity(chain_vec[i_rvec, :].reshape(-1, 3), unit=unit.angstrom))

        dcd = DCDFile(open(f'./ms0_car/fig_omm/c12pep_c36m/snake_path/snake_path_{save_tag}_min.dcd', 'wb'), psf.topology, .1)
        
        for i_rvec in range(chain_vec.shape[0]):
            dcd.writeModel(OmmQuantity(lowest_max_ener_cvec[i_rvec, :].reshape(-1, 3), unit=unit.angstrom))

        numpy.save(f'./ms0_car/fig_omm/c12pep_c36m/snake_path/snake_path_{save_tag}_min_ener.npy', ener)


    return chain.get_rep_list(), rms, ener, abslen


def snake_equi(rep_list: List[RepOpenMM], i_decoy) -> List[RepOpenMM]:
    """Greedy snake: Grow step. 
    Take the consumed list of replicas, optimize it. 
    """
    rep_list, rms, ener, abslen = exec_cos(str(i_decoy), rep_list)
    ener_str = str(ener).replace('\n', ' ')

    print(f'Grown.\n RMS: {numpy.mean(rms)}\nGrown ENER: {ener}\nGrown Head-Tail Length: {abslen}')
    log_out.write(f'----------Grown.\n  RMS: {numpy.mean(rms)}\n  ENER: {ener_str}\n  Head-Tail Length: {abslen}\n  Highest Energy: {numpy.max(ener[:-1])}\n\n')
    log_out.flush()

    return rep_list


if __name__ == '__main__':

    snake_list = []

    cvec = numpy.load('./ms0_car/fig_omm/c12pep_c36m/snake_path/snake_path_99.npy')
    
    for _ in range(cvec.shape[0]):
        snake_list.append(mk_rep(cvec[_, :]))

    snake = snake_equi(snake_list, 'fin')