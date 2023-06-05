# Integrated test for Muller Potential. 
# Zilin Song, 5 Jul 2022
# 

import sys, os



sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

from typing import Type
from pycospath.cos import ReaxPathCons as rpc, ConstAdvRep as CAR,             \
                          StringMethod as sm,  StringMethodGradProj as smgp,   \
                          CosBase, ReaxPathConsGradProj as rpcgp,              \
                          ConstAdvRepParamTan as carpt
from pycospath.chain        import Chain2D
from pycospath.comms.twodim import Rep2D, PotMuller

from pycospath.opt import ConsGradientDescent

import numpy

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def mk_plot(ax: Axes, cos: CosBase):
    """Plot the traj on MullerPot for one cos method."""

    # Control variables. 

    # cons_curv_thresh = 30

    n_steps = 100
    n_reps = 21 # rpc (but not rpcgp) is prone to fail with higher replica numbers. 
    xs = numpy.zeros((n_reps, ))
    ys = numpy.linspace(0, 2, num=n_reps)

    # Plot the Muller PES.
    pot = PotMuller()
    x, y, v = pot.get_pes()
    ax.contourf(x, y, v, 200)

    # set initial chain2d and cos
    rep_list = []

    for i_rep in range(n_reps):
        coor = numpy.asarray([xs[i_rep], ys[i_rep]])
        rep = Rep2D(coor, pot)
        rep_list.append(rep)
        
    chain = Chain2D(rep_list)

    cos = cos(chain)

    opt = ConsGradientDescent(cos, fix_mode='none')

    # Initial trajectory
    cvec = chain.get_chain_vec()
    # ax.scatter(cvec[:, 0], cvec[:, 1], s=0.5, c='blue', zorder=5)

    # Evolve the cos
    for _ in range(n_steps):
        opt.step()

    # Final trajectory
    cvec = cos.get_chain().get_chain_vec()
    ax.scatter(cvec[:, 0], cvec[:, 1], s=0.1, c='red', zorder=5)
    rms = cos.get_chain().get_rms(cos.get_chain().get_chain_vec())
    print(rms)
    print(numpy.mean(rms))

    # Set ticks
    ax.xaxis.set_ticks([-1.5, -.8, -.1,  .6, 1.3])
    ax.yaxis.set_ticks([-.5,   .2,  .9, 1.6, 2.3])
    ax.set_xlim(-1.5, 1.3)
    ax.set_ylim(-0.5, 2.3)

    ax.tick_params(direction='in')

    return ax
    
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9,6), dpi=300)

ax_i = mk_plot(ax[0, 0], CAR)
ax_i.set_title("Constant Advancement Replicas")

ax_i = mk_plot(ax[1, 0], carpt)
ax_i.set_title("Constant Advancement Replicas\nw/ Parametric Tangent")

ax_i = mk_plot(ax[0, 1], rpc)
ax_i.set_title("Reaction Path with\nHolonomic Constraints")

ax_i = mk_plot(ax[1, 1], rpcgp)
ax_i.set_title("Reaction Path with\nHolonomic Constraints & Grad Proj")

ax_i = mk_plot(ax[0, 2], sm)
ax_i.set_title("Simplified String Method")

ax_i = mk_plot(ax[1, 2], smgp)
ax_i.set_title("String Method w/ Grad Proj")


plt.tight_layout()
plt.savefig('./examples/2d_muller/intgr_test_results/intgr_test_muller_sd.png')
