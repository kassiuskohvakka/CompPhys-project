import pygmsh as pg 
import numpy as np
from matplotlib import pyplot as plt
import quadpy

from mpl_toolkits.mplot3d import Axes3D

import datetime
import time

from scipy import linalg as la

import argparse
#import matplotlib.lines.Line2D as Line2D



##################################
### PARSE ARGUMENTS
##################################

parser = argparse.ArgumentParser()

parser.add_argument('-s', type=int, default=0, help='Set to 1 to save the result in /data/')
parser.add_argument('-m', type=float, default=0.1, help='The mesh parameter. Defaults to 0.1.')

args = parser.parse_args()

SAVE_TO_FILE = args.s

##################################



# A function to calculate the area of a triangle given the coordinates of the vertices. 
# Needed in the construction of the FEM basis functions
def triangle_area(tr):
    return 0.5*abs( (tr[0][0] - tr[2][0])*(tr[1][1] - tr[0][1]) - (tr[0][0] - tr[1][0])*(tr[2][1] - tr[0][1]) )

# Test function for the 2D quadrature routine
#def f(x):
#    return np.sin(x[0]) * np.sin([x[1]])

# The potential function is our [0, 1]x[0, 1] domain
def pot(x):
    #print("x from pot function : {}".format(x))
    # if (x[0]==0 or x[0]==1 or x[1]==0 or x[1]==1):
    #     return 999999999
    # else:
    #     return 0

    # The input arrays must be numpy arrays!
    x_temp = np.transpose(np.array([x[0].flatten(), x[1].flatten()])) # x-coordinates
    # y_temp = x[1].flatten() # y-coordinates


    barrier_width = 0.03
    res = 9999*np.ones(x_temp.shape[0])
    for i, y in enumerate(x_temp):
        if (abs(y[0]-0)>barrier_width and abs(y[1]-0)>barrier_width and abs(y[0]-1)>barrier_width and abs(y[1]-1)>barrier_width):
            res[i] = -300*np.exp(-50*((y[0]-0.5)**2 + (y[1]-0.5)**2))

    return res


def FEM_hat(triangle, corner, x):
    if corner not in [0, 1, 2]:
        raise Exception('There are only three corners in a triangle!')

    return (1/(2*triangle_area(triangle)))*( triangle[(corner + 1)%3][0] * triangle[(corner + 2)%3][1] - triangle[(corner + 2)%3][0] * triangle[(corner + 1)%3][1] +
                                             (triangle[(corner + 1)%3][1] - triangle[(corner + 2)%3][1]) * x[0] +
                                             (triangle[(corner + 2)%3][0] - triangle[(corner + 1)%3][0]) * x[1])


def FEM_hat_grad(triangle, corner, corner2, x):
    if corner not in [0, 1, 2]:
        raise Exception('There are only three corners in a triangle!')
    y = np.ones(x.shape[1])

    return y*(1/(2*triangle_area(triangle)))**2 * ( (triangle[(corner + 1)%3][1] - triangle[(corner + 2)%3][1])*(triangle[(corner2+ 1)%3][1] - triangle[(corner2 + 2)%3][1]) +
                                                                  (triangle[(corner + 2)%3][0] - triangle[(corner + 1)%3][0]) * (triangle[(corner2 + 2)%3][0] - triangle[(corner2 + 1)%3][0]) )





###########################################################
### TRIANGULATE
###########################################################

# 0.02 is doable
mesh_parameter = args.m


geom = pg.built_in.Geometry()

# Define a simple 2D geometry, a unit square [0,1]x[0,1]

# Corner points
p0 = geom.add_point([0.0, 0.0, 0.0], lcar=mesh_parameter)
p1 = geom.add_point([1.0, 0.0, 0.0], lcar=mesh_parameter)
p2 = geom.add_point([1.0, 1.0, 0.0], lcar=mesh_parameter)
p3 = geom.add_point([0.0, 1.0, 0.0], lcar=mesh_parameter)

# p1 = pg.built_in.point.Point([1.0, 0.0, 0.0])
# p2 = pg.built_in.point.Point([1.0, 1.0, 0.0])
# p3 = pg.built_in.point.Point([0.0, 1.0, 0.0])

# Lines connecting the points
# l0 = pg.built_in.line.Line(p0, p1)
# l1 = pg.built_in.line.Line(p1, p2)
# l2 = pg.built_in.line.Line(p2, p3)
# l3 = pg.built_in.line.Line(p3, p0)

l0 = geom.add_line(p0, p1)
l1 = geom.add_line(p1, p2)
l2 = geom.add_line(p2, p3)
l3 = geom.add_line(p3, p0)


lineloop = geom.add_line_loop([l0, l1, l2, l3])

plane_surf = geom.add_plane_surface(lineloop, holes=None)

axis = [0, 0, 1]

mesh = pg.generate_mesh(geom)

# print(mesh.cells['triangle'])
# print('***')
# print(mesh.points)

# Get a easier handle on the coordinates of the triangles, albeit at the expense of data redundancy
triangle_coords = []

for tr in mesh.cells['triangle']:
	triangle_coords.append( [ 	[ mesh.points[tr[0]][0], mesh.points[tr[0]][1] ],
								[ mesh.points[tr[1]][0], mesh.points[tr[1]][1] ],
								[ mesh.points[tr[2]][0], mesh.points[tr[2]][1] ] ])

# print(triangle_coords)
triangle_coords = np.array(triangle_coords)


### TESTING
#triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
# Integrate a function over a triangle - Works.
# print(quadpy.triangle.integrate(f, triangle, quadpy.triangle.Strang(9)))


# Number of points
N = len(mesh.points)

# Kinetic matrix
T = np.zeros((N,N))

# Potential matrix
V = np.zeros((N,N))

# Overlap matrix
S = np.zeros((N,N))


# Start constructing the overlap matrix

for i, tr in enumerate(mesh.cells['triangle']):

    overlap00 = quadpy.triangle.integrate( lambda x: FEM_hat(triangle_coords[i], 0, x)*FEM_hat(triangle_coords[i], 0, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )
    overlap11 = quadpy.triangle.integrate( lambda x: FEM_hat(triangle_coords[i], 1, x)*FEM_hat(triangle_coords[i], 1, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )
    overlap22 = quadpy.triangle.integrate( lambda x: FEM_hat(triangle_coords[i], 2, x)*FEM_hat(triangle_coords[i], 2, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )


    overlap01 = quadpy.triangle.integrate( lambda x: FEM_hat(triangle_coords[i], 0, x)*FEM_hat(triangle_coords[i], 1, x),
    										triangle_coords[i], quadpy.triangle.Strang(9) )
    overlap02 = quadpy.triangle.integrate( lambda x: FEM_hat(triangle_coords[i], 0, x)*FEM_hat(triangle_coords[i], 2, x),
    										triangle_coords[i], quadpy.triangle.Strang(9) )
    overlap12 = quadpy.triangle.integrate( lambda x: FEM_hat(triangle_coords[i], 2, x)*FEM_hat(triangle_coords[i], 2, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )

    S[tr[0], tr[1]] += overlap01
    S[tr[1], tr[0]] += overlap01

    S[tr[0], tr[2]] += overlap02
    S[tr[2], tr[0]] += overlap02

    S[tr[1], tr[2]] += overlap12
    S[tr[2], tr[1]] += overlap12

    S[tr[0], tr[0]] += overlap00
    S[tr[1], tr[1]] += overlap11
    S[tr[2], tr[2]] += overlap22

# print(S)

# Start constructing the potential matrix

for i, tr in enumerate(mesh.cells['triangle']):

    pot00 = quadpy.triangle.integrate( lambda x: pot(x)*FEM_hat(triangle_coords[i], 0, x)*FEM_hat(triangle_coords[i], 0, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )
    pot11 = quadpy.triangle.integrate( lambda x: pot(x)*FEM_hat(triangle_coords[i], 1, x)*FEM_hat(triangle_coords[i], 1, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )
    pot22 = quadpy.triangle.integrate( lambda x: pot(x)*FEM_hat(triangle_coords[i], 2, x)*FEM_hat(triangle_coords[i], 2, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )


    pot01 = quadpy.triangle.integrate( lambda x: pot(x)*FEM_hat(triangle_coords[i], 0, x)*FEM_hat(triangle_coords[i], 1, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )
    pot02 = quadpy.triangle.integrate( lambda x: pot(x)*FEM_hat(triangle_coords[i], 0, x)*FEM_hat(triangle_coords[i], 2, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )
    pot12 = quadpy.triangle.integrate( lambda x: pot(x)*FEM_hat(triangle_coords[i], 2, x)*FEM_hat(triangle_coords[i], 2, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )

    V[tr[0], tr[1]] += pot01
    V[tr[1], tr[0]] += pot01

    V[tr[0], tr[2]] += pot02
    V[tr[2], tr[0]] += pot02

    V[tr[1], tr[2]] += pot12
    V[tr[2], tr[1]] += pot12

    V[tr[0], tr[0]] += pot00
    V[tr[1], tr[1]] += pot11
    V[tr[2], tr[2]] += pot22


# Start constructing the potential matrix

for i, tr in enumerate(mesh.cells['triangle']):

    kin00 = quadpy.triangle.integrate( lambda x: FEM_hat_grad(triangle_coords[i], 0, 0, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )
    kin11 = quadpy.triangle.integrate( lambda x: FEM_hat_grad(triangle_coords[i], 1, 1, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )
    kin22 = quadpy.triangle.integrate( lambda x: FEM_hat_grad(triangle_coords[i], 2, 2, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )


    kin01 = quadpy.triangle.integrate( lambda x: FEM_hat_grad(triangle_coords[i], 0, 1, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )
    kin02 = quadpy.triangle.integrate( lambda x: FEM_hat_grad(triangle_coords[i], 0, 2, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )
    kin12 = quadpy.triangle.integrate( lambda x: FEM_hat_grad(triangle_coords[i], 1, 2, x),
                                            triangle_coords[i], quadpy.triangle.Strang(9) )

    T[tr[0], tr[1]] += kin01
    T[tr[1], tr[0]] += kin01

    T[tr[0], tr[2]] += kin02
    T[tr[2], tr[0]] += kin02

    T[tr[1], tr[2]] += kin12
    T[tr[2], tr[1]] += kin12

    T[tr[0], tr[0]] += kin00
    T[tr[1], tr[1]] += kin11
    T[tr[2], tr[2]] += kin22

# print(V)

# Construct the Hamiltonian

H = 0.5*T + V

E, psi = la.eigh(H[1:-1, 1:-1], b=S[1:-1, 1:-1])


psi = np.vstack((np.zeros((1, N-2)), psi, np.zeros((1, N-2))))

ts = time.time()
readts = datetime.datetime.fromtimestamp(round(ts)).isoformat()

filename = "data/results_m{}_{}".format(mesh_parameter, readts)

data = np.array([E, psi])

np.save(filename, data)




### Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # plt.plot(psi[:,0]
# ax.plot_trisurf(mesh.points[:,0], mesh.points[:,1], psi[:,1])
# plt.show()










### Visually check that triangulation was successful
# plt.figure()
# plt.scatter(mesh.points[:,0], mesh.points[:,1])
# # for tr in mesh.cells['triangle']:
# #   plt.plot([mesh.points[tr[0]][0] , mesh.points[tr[1]][0]], [mesh.points[tr[0]][1], mesh.points[tr[1]][1]])
# #   plt.plot([mesh.points[tr[1]][0] , mesh.points[tr[2]][0]], [mesh.points[tr[1]][1], mesh.points[tr[2]][1]])
# #   plt.plot([mesh.points[tr[2]][0] , mesh.points[tr[0]][0]], [mesh.points[tr[2]][1], mesh.points[tr[0]][1]])

# X, Y = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
# # print(X.shape)
# Z = pot(np.array([X,Y]))
# plt.contourf(X,Y, np.reshape(Z, X.shape))
# plt.show()





