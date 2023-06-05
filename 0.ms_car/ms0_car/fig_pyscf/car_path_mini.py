# Test on Baker set reactions. 
# Zilin Song, 08 Apr 2023
# 

import sys, os
from typing import Tuple, List

sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd()) # to permit access to pycospath.

# pycospath
from pycospath.cos   import ConstAdvRepParamTan as CARPT
from pycospath.utils import rigid_fit
from pycospath.chain import ChainCart
from pycospath.opt   import ConsGradientDescent
from pycospath.comms.pyscf import RepPySCF

import numpy

xyzdir = "./ms0_car/fig_pyscf/end_points"
outdir = "./ms0_car/fig_pyscf/car_paths"
  
pyscf_xc         = 'b3lyp'
pyscf_mole_basis = '6-31g**'
pyscf_mole_spin  = 0

def read_xyz(file_dir: str):
    assert not file_dir is None
    
    atom_list   = []
    coordinates = []

    with open(file_dir, 'r') as fi:
        lines = fi.readlines()

        for l in lines[2:]:
            words = l.strip().split()
            
            if len(words) == 4:
                atom_list.append(words[0])
                coordinates.append([float(words[1]), float(words[2]), float(words[3])])
    
    return atom_list, numpy.asarray(coordinates).flatten()

def write_xyz_traj(fname: str, chain_vec, atom_names):
    """Output frames to xyz file."""
    with open(fname, 'w') as fo:

        for i in range(chain_vec.shape[0]):
            fo.write(f'{len(atom_names)}\n')
            fo.write(f'XYZ coordinates of replica {i}.\n')

            rep_vec = chain_vec[i]

            # rep_vec = rep_vec.reshape((-1, 3))
            assert rep_vec.shape == (len(atom_names)*3, )

            for row in range(len(atom_names)):
                fo.write(f'{atom_names[row]:<8}{rep_vec[row*3]:<16.8f}{rep_vec[row*3+1]:<16.8f}{rep_vec[row*3+2]:<16.8f}\n')

def mk_rep(atom_names, rep_vec: numpy.ndarray, pyscf_mole_chrg) -> RepPySCF:
    """Make a PySCF replica from the molecule topology and replica vector."""
    assert rep_vec.shape == (len(atom_names)*3, )
    
    mole_str = ""
    
    for i in range(len(atom_names)):
        mole_str += \
f'''{atom_names[i]:<4}{rep_vec[i*3]:<16.8f}{rep_vec[i*3+1]:<16.8f}{rep_vec[i*3+2]:<16.8f}
'''
    print(mole_str)
    return RepPySCF(pyscf_xc=pyscf_xc, 
                    pyscf_mole_atoms=mole_str, 
                    pyscf_mole_basis=pyscf_mole_basis, 
                    pyscf_mole_spin=0,
                    pyscf_mole_chrg=pyscf_mole_chrg, 
                    is_mass_weighted=False)

def mk_chain(rxn_id: int) -> Tuple[ChainCart, List]:
    """Get intepolated initial path as a ChainCart."""
    if rxn_id in [1, 11, 14, 15, 21, 23]:
        with_transcient = True
    else:
        with_transcient = False
    
    chrg = -1 if rxn_id == 16 else 1 if rxn_id == 20 else 0
    
    reactant_xyz = f'{xyzdir}/rxn{rxn_id}/reactant.xyz'
    product_xyz  = f'{xyzdir}/rxn{rxn_id}/product.xyz'

    reactant_atom_names, reactant_rep_vec = read_xyz(reactant_xyz)
    product_atom_names,  product_rep_vec  = read_xyz(product_xyz)
    
    assert reactant_atom_names == product_atom_names
    
    rep_list = []

    if with_transcient == False:
        product_rep_vec, _, _ = rigid_fit(reactant_rep_vec, product_rep_vec)
        
        for i in range(7):
            rep_vec = reactant_rep_vec + float(i/6) * (product_rep_vec - reactant_rep_vec)
            rep = mk_rep(reactant_atom_names, rep_vec, chrg)
            rep_list.append(rep)
    
    else:
        transcient_xyz = f'{xyzdir}/rxn{rxn_id}/transcient.xyz'
        transcient_atom_names, transcient_rep_vec = read_xyz(transcient_xyz)
        
        assert transcient_atom_names == reactant_atom_names

        transcient_rep_vec, _, _ = rigid_fit(reactant_rep_vec, transcient_rep_vec)
        for i in range(4):
            rep_vec = reactant_rep_vec + float(i/3) * (transcient_rep_vec - reactant_rep_vec)
            rep = mk_rep(reactant_atom_names, rep_vec, chrg)
            rep_list.append(rep)

        product_rep_vec, _, _ = rigid_fit(transcient_rep_vec, product_rep_vec)
        for i in range(1, 4):
            rep_vec = transcient_rep_vec + float(i/3) * (product_rep_vec - transcient_rep_vec)
            rep = mk_rep(reactant_atom_names, rep_vec, chrg)
            rep_list.append(rep)
        
    assert len(rep_list) == 7

    return ChainCart(rep_list), reactant_atom_names

def exec_cos(chain: ChainCart, atom_names, work_dir: str):
    """Perform CAR-PT calculations."""
    cos = CARPT(chain, verbose=False, cons_log_writer=open(f'{work_dir}/mini_cons.log', 'w'))
    opt = ConsGradientDescent(cos, fix_mode='both')

    # Warm init the optimizer.
    pv_cvec, (pv_ener, pv_grad) = chain.get_chain_vec(), chain.get_eners_grads()
    opt._x_pv, opt._y_pv, opt._dy_pv = pv_cvec, pv_ener, pv_grad
    opt._init_step = True
    
    # loggers.
    run_logger = open(f'{work_dir}/mini_ener.log', 'w')

    # output init ener.
    current_total_ene = numpy.sum(pv_ener - pv_ener[0])
    current_diffr_ene = 0.
    current_diffr_cor = 0.
    current_strng_ene = str(pv_ener-pv_ener[0])[1:-1].replace('\n', ' ')
    
    run_logger.write(f'##    0 :lr {opt._gd_eta:8.6f} :tot_ene {current_total_ene:>10.4f} :dif_ene {current_diffr_ene:>10.4f} :dif_cor {current_diffr_cor:>10.4f} : {current_strng_ene}\n')
    run_logger.flush()

    for istep in range(1, 501):
        # Previous state.
        opt_previous_state = opt._x_pv.copy(), opt._y_pv.copy(), opt._dy_pv.copy()
        step_rejected = False

        # Take 1 step.
        opt.step()

        # Current state.
        opt_current_state = opt._x_pv.copy(), opt._y_pv.copy(), opt._dy_pv.copy()

        # Stats.
        current_total_ene = numpy.sum(opt_current_state[1] - opt_current_state[1][0])
        current_diffr_ene = numpy.sum(numpy.abs(opt_current_state[1] - opt_previous_state[1]))
        current_diffr_cor = numpy.sum(numpy.abs(opt_current_state[0] - opt_previous_state[0]))
        current_strng_ene = str(opt_current_state[1] - opt_current_state[1][0])[1:-1].replace('\n', ' ') # starts/ends with []

        if current_diffr_ene <= 0.07: # Convergence criteria.
            numpy.save(f'{work_dir}/mini_finl.npy', chain.get_chain_vec())
            run_logger.write(f'!! cvrg :lr {opt._gd_eta:8.6f} :tot_ene {current_total_ene:>10.4f} :dif_ene {current_diffr_ene:>10.4f} :dif_cor {current_diffr_cor:>10.4f} : {current_strng_ene}\n')
            break
        
        else: # not converged.
            
            if numpy.sum(opt_current_state[1]) <= numpy.sum(opt_previous_state[1]): 
                # Path energy decreased. 
                opt._gd_eta *= 1.2

            else:
                
                if numpy.max(opt_current_state[1]) > numpy.max(opt_previous_state[1]):
                    # Path energy increased, TS energy increased -> Halfen LR and reject step.
                    step_rejected = True
                    opt._gd_eta *= .5
                    opt._x_pv, opt._y_pv, opt._dy_pv = opt_previous_state
                    chain.set_chain_vec(opt_current_state[0])

                else:
                    # Path energy increased, TS energy decreased.
                    opt._gd_eta *= .5
        
        istep_str = 'rjct' if step_rejected == True else str(istep)
        run_logger.write(f'## {istep_str:>4} :lr {opt._gd_eta:8.6f} :tot_ene {current_total_ene:>10.4f} :dif_ene {current_diffr_ene:>10.4f} :dif_cor {current_diffr_cor:>10.4f} : {current_strng_ene}\n')
        run_logger.flush()

    write_xyz_traj(f'{work_dir}/mini_finl.xyz', chain.get_chain_vec(), atom_names)
    
if __name__ == "__main__":
    rxn_id = int(sys.argv[1])

    chain, atom_names = mk_chain(rxn_id)

    exec_cos(chain, atom_names, f'{outdir}/rxn{rxn_id}')