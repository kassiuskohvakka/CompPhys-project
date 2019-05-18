import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc

from mpl_toolkits.mplot3d import Axes3D


####################################
### MATPLOTLIB SETTINGS
####################################

# plt.ion()

def mpl_settings( axlabelsize=12, ticksize=8):
    rc('font', **{'family' : 'serif'})
    rc('text', usetex=True)

    matplotlib.rcParams.update({'font.size' : ticksize})
    matplotlib.rcParams.update({'axes.labelsize' : axlabelsize})
    matplotlib.rcParams.update({'xtick.labelsize' : ticksize})
    matplotlib.rcParams.update({'ytick.labelsize' : ticksize})
    matplotlib.rcParams.update({'legend.fontsize' : ticksize})
    matplotlib.rcParams.update({'lines.linewidth' : 2})



####################################
### READ DATA FROM FILE 
####################################


# filename = 'data/results_m0.05_2019-05-09T11:09:46.npy'
# filename = 'data/results_m0.05_2019-05-09T14:00:24.npy'
filename = 'data/circ_pot_test.npy'
# filename = "data/test.npy"

data = np.load(filename, allow_pickle=True)


# Energy eigenvalues
E = data[0]

# Eigenstates
psi = data[1]

# Mesh points
mesh_points = data[2]

# Potential landscape
pot = data[3]
X, Y = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))

# Dimension
N = data[4]

print(N)



##################################
### TRISURF PLOT
##################################



mpl_settings(12, 8)
f1 = plt.figure(figsize=(8,12), dpi=80)
ax0 = f1.add_subplot(321, projection='3d')
ax1 = f1.add_subplot(322, projection='3d')
ax2 = f1.add_subplot(323, projection='3d')
ax3 = f1.add_subplot(324, projection='3d')
ax4 = f1.add_subplot(325, projection='3d')
ax5 = f1.add_subplot(326, projection='3d')

ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')
ax0.set_zlabel('$\psi_0(x,y)$')
ax0.set_title(f'$E_0 = {round(E[0],1)}$')
ax0.view_init(elev=15, azim=-30)
ax0.plot_trisurf(mesh_points[:,0], mesh_points[:,1], -psi[:,0], cmap='plasma', shade=True)
ax0.grid(False)

ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_zlabel('$\psi_1(x,y)$')
ax1.set_title(f'$E_1 = {round(E[1],1)}$')
ax1.view_init(elev=15, azim=-30)
ax1.plot_trisurf(mesh_points[:,0], mesh_points[:,1], psi[:,1], cmap='plasma', shade=True)
ax1.grid(False)

ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
ax2.set_zlabel('$\psi_2(x,y)$')
ax2.set_title(f'$E_2 = {round(E[2],1)}$')
ax2.view_init(elev=15, azim=-30)
ax2.plot_trisurf(mesh_points[:,0], mesh_points[:,1], psi[:,2], cmap='plasma', shade=True)
ax2.grid(False)

ax3.set_xlabel('$x$')
ax3.set_ylabel('$y$')
ax3.set_zlabel('$\psi_3(x,y)$')
ax3.set_title(f'$E_3 = {round(E[3],1)}$')
ax3.view_init(elev=15, azim=-30)
ax3.plot_trisurf(mesh_points[:,0], mesh_points[:,1], psi[:,3], cmap='plasma', shade=True)
ax3.grid(False)

ax4.set_xlabel('$x$')
ax4.set_ylabel('$y$')
ax4.set_zlabel('$\psi_4(x,y)$')
ax4.set_title(f'$E_3 = {round(E[4],1)}$')
ax4.view_init(elev=15, azim=-30)
ax4.plot_trisurf(mesh_points[:,0], mesh_points[:,1], psi[:,4], cmap='plasma', shade=True)
ax4.grid(False)

ax5.set_xlabel('$x$')
ax5.set_ylabel('$y$')
ax5.set_zlabel('$\psi_5(x,y)$')
ax5.set_title(f'$E_5 = {round(E[5],1)}$')
ax5.view_init(elev=15, azim=-30)
ax5.plot_trisurf(mesh_points[:,0], mesh_points[:,1], psi[:,5], cmap='plasma', shade=True)
ax5.grid(False)




plt.show()


f1.savefig("figs/spam.pdf")

########################
### POTENTIAL PLOT
########################


## Visually check that triangulation was successful
mpl_settings(20, 18)
f2 = plt.figure(figsize=(5,5), dpi=80)
# for tr in mesh.cells['triangle']:
#   plt.plot([mesh.points[tr[0]][0] , mesh.points[tr[1]][0]], [mesh.points[tr[0]][1], mesh.points[tr[1]][1]])
#   plt.plot([mesh.points[tr[1]][0] , mesh.points[tr[2]][0]], [mesh.points[tr[1]][1], mesh.points[tr[2]][1]])
#   plt.plot([mesh.points[tr[2]][0] , mesh.points[tr[0]][0]], [mesh.points[tr[2]][1], mesh.points[tr[0]][1]])
trunc_pot = np.array([min(p, abs(min(pot))) for p in pot])
trunc_pot = np.reshape(trunc_pot, X.shape)

plt.contourf(X,Y, trunc_pot, cmap='bwr', levels=20)
plt.scatter(mesh_points[:,0], mesh_points[:,1], s=0.7)

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
plt.show()
f2.savefig("figs/eggs.pdf")












