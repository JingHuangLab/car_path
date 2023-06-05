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

log_out = open('./ms0_car/fig_omm/c12pep_c36m/greedy_snake.log', 'w')

def mk_rep(rep_vec: numpy.ndarray) -> RepOpenMM:
    """Make a RepOpenMM object with the given rep_vec.
    rep_vec is first transformed to an OmmQuantity object which is used to init
    the RepOpenMM object.
    """
    rep_coor = OmmQuantity(rep_vec.reshape(-1, 3), unit=unit.angstrom)
    rep = RepOpenMM(rep_coor, omm_pot, mscale_vec = mscale_vec) # H atoms does not contribute to RMS. 

    return rep

def mk_decoy_rep_list(n_reps: int = 100) -> ChainCart: 
    """Make a list of replicas from the initial guessed / decoy replicas. """
    ref_pdb = PDBFile("./examples/omm_c36m_c12pep/init_guess/r0.pdb")
    ref_rep_vec = numpy.asarray(ref_pdb.positions.value_in_unit(unit.angstrom)).flatten()

    rep_list = []

    for i_rep in range(n_reps):
        pdb = PDBFile(f"./examples/omm_c36m_c12pep/init_guess/r{i_rep}.pdb")
        rep_vec = numpy.asarray(pdb.positions.value_in_unit(unit.angstrom)).flatten()
        rep_vec_aligned, _, _ = rigid_fit(ref_rep_vec, rep_vec)

        rep = mk_rep(rep_vec_aligned)
        rep_list.append(rep)

    return rep_list

def exec_cos(save_tag: int,
             rep_list:     List[RepOpenMM], 
             adam_n_steps: int = 0, 
             gd_n_steps:   int = 0, 
             fix_mode:     str = 'none'
             ) -> List[RepOpenMM]: 
    """Execute chain-of-states optimizations with kwarg specs."""
    chain = ChainCart(rep_list)
    cos = CAR(chain, verbose=False, cons_curvreglr_thresh=15)

    opt_adam = ConsAdaptiveMomentum(cos, fix_mode = fix_mode)

    for _ in range(adam_n_steps):
        
        if _ % 100 == 1:
            print(f'  AdaM step: {_-1} / {adam_n_steps}')
            log_out.write(f'  AdaM step: {_-1} / {adam_n_steps}\n')
            log_out.flush()
        
        opt_adam.step()
    
    opt_grad = ConsGradientDescent(cos, fix_mode = fix_mode)

    for _ in range(gd_n_steps):
        
        if _ % 10 == 1:
            print(f'  GraD step: {_-1} / {gd_n_steps}')
            log_out.write(f'  GraD step: {_-1} / {gd_n_steps}\n')
            log_out.flush()

        opt_grad.step()

    chain_vec = chain.get_chain_vec()

    if save_tag != 'nosave':
        numpy.save(f'./ms0_car/fig_omm/c12pep_c36m/snake_path/snake_path_{save_tag}.npy', chain_vec)

        dcd = DCDFile(open(f'./ms0_car/fig_omm/c12pep_c36m/snake_path/snake_path_{save_tag}.dcd', 'wb'), psf.topology, .1)
        
        for i_rvec in range(chain_vec.shape[0]):
            dcd.writeModel(OmmQuantity(chain_vec[i_rvec, :].reshape(-1, 3), unit=unit.angstrom))

    abslen = numpy.linalg.norm(chain_vec[-1] - chain_vec[0])
    rms    = chain.get_rms(chain_vec)
    ener   = chain.get_eners_grads()[0]

    return chain.get_rep_list(), rms, ener, abslen

def snake_dine_consume(rep_list: List[RepOpenMM]) -> List[RepOpenMM]:
    """Greedy Snake: Dining step. 
    Keeps adding new replicas at the mid-point of the last two replicas, until 
    the optimized RMS distances of the newly added replicas were lower than a 
    pre-defined threshold. 
    """
    ref_rep = rep_list[0]
    ini_rep = rep_list[-2]
    fin_rep = rep_list[-1]

    ref_rep_vec = ref_rep.get_rep_vec()

    tmp_chain = ChainCart(rep_list)
    tmp_rms   = tmp_chain.get_rms(tmp_chain.get_chain_vec())
    n_rep_add = int(numpy.mean(tmp_rms) / thresh_consume_rms) + 1

    consume_list = [ini_rep, fin_rep]

    for i_rep in range(1, n_rep_add):
        ini_rep_vec = ini_rep.get_rep_vec()
        fin_rep_vec = fin_rep.get_rep_vec()
        ini_rep_vec_aligned, _, _ = rigid_fit(ref_rep_vec, ini_rep_vec)
        fin_rep_vec_aligned, _, _ = rigid_fit(ref_rep_vec, fin_rep_vec)

        add_rep_vec = ini_rep_vec_aligned + float(i_rep / n_rep_add) * (fin_rep_vec_aligned - ini_rep_vec_aligned)
        add_rep = mk_rep(add_rep_vec)

        consume_list.insert(-1, add_rep)
    
    print(f' Total reps: {len(snake_list)}')
    log_out.write(f' Total reps: {len(snake_list)}\n')
    log_out.flush()
    
    """Greedy Snake: Consume step. 
    Briefly optimize the dined sub-chain-of-replicas. 
    """

    consume_list, rms, ener, abslen = exec_cos('nosave', consume_list, 300, 30, 'both')
    ener_str = str(ener).replace('\n', ' ')

    print(f'Consumed.\n head-tail dist: {abslen}\n ener: {ener_str}')
    log_out.write(f'----------Consumed.\n  head-tail dist: {abslen}\n  ener: {ener_str}\n')
    log_out.flush()

    # pop the head and the decoy from the rep_list, and re-add them from consume_list
    rep_list.pop(-1)
    rep_list.pop(-1)

    for rep in consume_list:
        rep_list.append(rep)

    return rep_list

def snake_grow(rep_list: List[RepOpenMM], i_decoy) -> List[RepOpenMM]:
    """Greedy snake: Grow step. 
    Take the consumed list of replicas, optimize it. 
    """
    # 'tail' is the last consumed replica, i.e., where the head of the snake at.
    # fix_mode in optimizer classes uses 'tail' to denote the last replica (here
    # the snake head). I know this looks strange lol. 
    
    rep_list, rms, ener, abslen = exec_cos(str(i_decoy), rep_list, 500, 30, 'tail')
    ener_str = str(ener).replace('\n', ' ')

    print(f'Grown.\n RMS: {numpy.mean(rms)}\nGrown ENER: {ener}\nGrown Head-Tail Length: {abslen}')
    log_out.write(f'----------Grown.\n  RMS: {numpy.mean(rms)}\n  ENER: {ener_str}\n  Head-Tail Length: {abslen}\n  Highest Energy: {numpy.max(ener[:-1])}\n\n')
    log_out.flush()

    return rep_list


if __name__ == '__main__':
    decoy_list = mk_decoy_rep_list()

    snake_list = []

    # cvec = numpy.load('./ms0_car/fig_omm/c12pep_c36m/snake_path/snake_path_64.npy')
    
    # for _ in range(cvec.shape[0]):
    #     snake_list.append(mk_rep(cvec[_, :]))

    for i_decoy in range(0, len(decoy_list)):
        print('\n==========\nDining...', i_decoy)
        log_out.write(f'\n==========\nDining... {str(i_decoy)}\n')
        log_out.flush()

        snake_list.append(decoy_list[i_decoy])

        if len(snake_list) < 5:
            continue
        
        else:
            print(f'Consuming...\n', flush=True)
            log_out.write('Consuming...\n')
            log_out.flush()


            snake_list = snake_dine_consume(snake_list)

            print('Growing...\n')
            log_out.write('Growing...\n')
            log_out.flush()

            snake_list = snake_grow(snake_list,  i_decoy)
