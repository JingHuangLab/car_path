# Plot the diala isomerization pes. 
# Ye Ding, 15 Jun 2022.

import numpy
import matplotlib.pyplot as plt

x = numpy.load('./examples/diala_mpid/omm_pes/mpid_phi.npy')
y = numpy.load('./examples/diala_mpid/omm_pes/mpid_psi.npy')
e = numpy.load('./examples/diala_mpid/omm_pes/mpid_ene.npy')

fig, ax = plt.subplots(dpi=300)

print(f'Ener range: min = {numpy.min(e)}; max = {numpy.max(e)}, ')

pes_fig = ax.contourf(x, y, e, levels=numpy.linspace(50, 85, 100), extend='both')
ene_bar = fig.colorbar(pes_fig, ticks=numpy.linspace(50, 85, 8))
ene_bar.ax.set_ylabel(r"Energy ($kcal mol^{-1})$")

import matplotlib.ticker as plt_ticker
loc = plt_ticker.MultipleLocator(base=30.0) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
plt.xlabel(r"$\Phi$ ($^\circ$)")
plt.ylabel(r"$\Psi$ ($^\circ$)")

plt.savefig('./examples/diala_mpid/omm_pes/mpid_pes.png')