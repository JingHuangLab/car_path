# Integrated test for Muller Potential. 
# Zilin Song, 5 Jul 2022
# 

import sys, os

sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

from typing import Type
from pycospath.cos import CosBase, ConstAdvRep as CAR,                         \
                                   ConstAdvRepParamTan as carpt,               \
                                   StringMethod as sm
from pycospath.chain        import Chain2D
from pycospath.comms.twodim import Rep2D, PotMuller
from pycospath.opt import ConsGradientDescent

import numpy

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def mk_plot(ax: Axes, cos: Type[CosBase]):
    """Plot the traj on MullerPot for one cos method."""

    # Control variables. 
    n_reps = 21
    n_steps = 200
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

    cos: CosBase = cos(chain)
    # cos: CosBase = cos(chain, cons_thresh=1e-15, cons_maxitr=100)

    opt = ConsGradientDescent(cos)

    # Initial trajectory
    cvec = chain.get_chain_vec()
    ax.scatter(cvec[:, 0], cvec[:, 1], s=0.5, c='blue', zorder=5)

    # Evolve the cos
    for _ in range(n_steps):
        opt.step()

    cvec = cos.get_chain().get_chain_vec()
    ax.plot(cvec[:, 0], cvec[:, 1], 'o-', lw=0.5, markersize=0.5, c='gray')

    # Final trajectory
    cvec = cos.get_chain().get_chain_vec()
    ax.scatter(cvec[:, 0], cvec[:, 1], s=0.5, c='red', zorder=5)

    # Set ticks
    ax.xaxis.set_ticks([-1.5, -.8, -.1,  .6, 1.3])
    ax.yaxis.set_ticks([-.5,   .2,  .9, 1.6, 2.3])
    ax.set_xlim(-1.5, 1.3)
    ax.set_ylim(-0.5, 2.3)

    ax.tick_params(direction='in')

    return ax
    
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6,6), dpi=300)

ax_i = mk_plot(ax[0, 0], CAR)
ax_i.set_title("Constant Advancement Replicas")

ax_i = mk_plot(ax[1, 0], carpt)
ax_i.set_title("Constant Advancement Replicas\nw/ Parametrized Tangent")

ax_i = mk_plot(ax[1, 1], sm)
ax_i.set_title("Simplified String Method")


plt.tight_layout()
plt.savefig('./examples/2d_muller/intgr_test_results/intgr_test_muller_tans.png')
