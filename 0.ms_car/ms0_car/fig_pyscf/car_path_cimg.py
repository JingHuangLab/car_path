# Test on Baker set reactions of closed shell systems. 
# Zilin Song, 10 Apr 2023.
# 

import sys, os
from typing import Tuple, List

sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd()) # to permit access to pycospath.

# pycospath
from pycospath.cos   import ConstAdvRepParamTan as CARPT, CimgConstAdvRep as CICAR
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
    chrg = -1 if rxn_id == 16 else 1 if rxn_id == 20 else 0
    
    reactant_xyz = f'{xyzdir}/rxn{rxn_id}/reactant.xyz'

    reactant_atom_names, reactant_rep_vec = read_xyz(reactant_xyz)
        
    rep_list = [mk_rep(reactant_atom_names, reactant_rep_vec, chrg) for i in range(7)]

    chain = ChainCart(rep_list)
    
    return chain, reactant_atom_names

def exec_cos(chain: ChainCart, atom_names, work_dir: str):
    """Perform CI-CAR-PT calculations."""
    # Read previously minimized path coordinates.
    chain.set_chain_vec(numpy.load(f'{work_dir}/mini_finl.npy'))
    cos = CICAR(CARPT(chain, verbose=False))
    opt = ConsGradientDescent(cos, gd_eta=0.01, gd_eta_scal=.98, fix_mode='both')

    # Warm init the optimizer.
    pv_cvec, (pv_ener, pv_grad) = chain.get_chain_vec(), chain.get_eners_grads()
    opt._x_pv, opt._y_pv, opt._dy_pv = pv_cvec, pv_ener, pv_grad
    opt._init_step = True

    if numpy.argmax(pv_ener) in [0, 1, 5, 6]:
        print("CIMG CANCELED.")
        return

    # loggers.
    run_logger = open(f'{work_dir}/cimg_ener.log', 'w')

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

        # Take 1 step.
        opt.step()

        # Current state.
        opt_current_state = opt._x_pv.copy(), opt._y_pv.copy(), opt._dy_pv.copy()

        # Stats.
        current_total_ene = numpy.sum(opt_current_state[1] - opt_current_state[1][0])
        current_diffr_ene = numpy.sum(numpy.abs(opt_current_state[1] - opt_previous_state[1]))
        current_diffr_cor = numpy.sum(numpy.abs(opt_current_state[0] - opt_previous_state[0]))
        current_strng_ene = str(opt_current_state[1] - opt_current_state[1][0])[1:-1].replace('\n', ' ') # starts/ends with []

        # Path energy decreased, TS energy decreased -> Upscale lr by 1.1.
        run_logger.write(f'## {istep:>4} :lr {opt._gd_eta:8.6f} :tot_ene {current_total_ene:>10.4f} :dif_ene {current_diffr_ene:>10.4f} :dif_cor {current_diffr_cor:>10.4f} : {current_strng_ene}\n')
        run_logger.flush()

        if current_diffr_ene <= 0.07: # Convergence criteria.
            numpy.save(f'{work_dir}/cimg_finl.npy', chain.get_chain_vec())
            run_logger.write(f'!! cvrg :lr {opt._gd_eta:8.6f} :tot_ene {current_total_ene:>10.4f} :dif_ene {current_diffr_ene:>10.4f} :dif_cor {current_diffr_cor:>10.4f} : {current_strng_ene}\n')
            break
        
    write_xyz_traj(f'{work_dir}/cimg_finl.xyz', chain.get_chain_vec(), atom_names)

if __name__ == "__main__":
    rxn_id = int(sys.argv[1])

    chain, atom_names = mk_chain(rxn_id)

    exec_cos(chain, atom_names, f'{outdir}/rxn{rxn_id}')