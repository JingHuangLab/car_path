# Integrated test for Muller Potential. 
# Zilin Song, 5 Jul 2022
# 

import sys, os

sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

from typing import Type
from pycospath.cos import CosBase, ConsCimgBase,                               \
                          ConstAdvRep as CAR,                                  \
                          ConstAdvRepParamTan as carpt,                        \
                          StringMethod as sm,                                  \
                          CimgConstAdvRep as ci_car,                           \
                          CimgStringMethod as ci_sm, CimgBase

from pycospath.opt          import ConsGradientDescent
from pycospath.chain        import Chain2D
from pycospath.comms.twodim import Rep2D, PotSymDoubleWell

import numpy

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def mk_plot(ax: Axes, cos: Type[CosBase], ci_cos: Type[ConsCimgBase]):
    """Plot the traj on MullerPot for one cos method."""

    # Control variables. 
    n_reps = 10
    n_steps = 200
    xs = numpy.linspace(-1.5, 1.5, num=n_reps) 
    ys = numpy.linspace(-1.5, 1.5, num=n_reps)  # zeros((n_reps, ))

    # Plot the Muller PES.
    pot = PotSymDoubleWell()
    x, y, v = pot.get_pes()
    ax.contourf(x, y, v, 200)

    ax.scatter([[0]], [[0]], s=5, c='cyan', zorder=4)

    # set initial chain2d and cos
    rep_list = []

    for i_rep in range(n_reps):
        coor = numpy.asarray([xs[i_rep], ys[i_rep]])
        rep = Rep2D(coor, pot)
        rep_list.append(rep)
        
    chain = Chain2D(rep_list)

    cos: CosBase = cos(chain)

    opt = ConsGradientDescent(cos)

    for _ in range(25):
        opt.step()

    ci_cos: CimgBase = ci_cos(cos)

    opt = ConsGradientDescent(ci_cos)

    for _ in range(n_steps):
        opt.step()

    cvec = ci_cos.get_chain().get_chain_vec()
    
    ax.plot(cvec[:, 0], cvec[:, 1], 'o-', lw=0.5, markersize=0.5, c='gray')

    # Final trajectory
    cvec = ci_cos.get_chain().get_chain_vec()
    ax.scatter(cvec[:, 0], cvec[:, 1], s=0.5, c='red', zorder=5)

    # Set ticks
    # ax.xaxis.set_ticks([-1.5, -.8, -.1,  .6, 1.3])
    # ax.yaxis.set_ticks([-.5,   .2,  .9, 1.6, 2.3])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    ax.tick_params(direction='in')

    return ax, cvec
    
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6,6), dpi=300)

ax_i, a = mk_plot(ax[0, 0], CAR, ci_car)
ax_i.set_title("Climbing Image\nNudegd Rigid Chains")

ax_i, c = mk_plot(ax[1, 0], carpt, ci_car)
ax_i.set_title("Climbing Image\nConstant Advancement Replicas\nw/ Parametrized Tangent")

ax_i, d = mk_plot(ax[1, 1], sm, ci_sm)
ax_i.set_title("Climbing Image\nString Method")

print(a[5])
print(c[4])
print(d[4])

# ax_i = mk_plot(ax[1, 1], sm)
# ax_i.set_title("Simplified String Method")


plt.tight_layout()
plt.savefig('./examples/2d_double_well/intgr_test_results/intgr_test_muller_tans_cimg.png')
