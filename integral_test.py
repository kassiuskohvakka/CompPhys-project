import pygmsh as pg 
import numpy as np
from matplotlib import pyplot as plt
import quadpy

from mpl_toolkits.mplot3d import Axes3D

import datetime
import time

import threading
import multiprocessing

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
mesh_parameter = args.m

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
            #Compute the distance from (0.5, 0.5)
            dist_sq = (y[0]-0.5)**2 + (y[1]-0.5)**2
            if (dist_sq>0.13):
                res[i] = 9999#0.0
            elif (dist_sq <0.04):
                res[i] = 9999#min(9999, 1/dist_sq)#1/(dist_sq)+0.001
            else:
                res[i] = 9999*(dist_sq-0.04)*(dist_sq-0.13)
            

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

print(f'Save results : {SAVE_TO_FILE}')
print(f'Mesh parameter : {mesh_parameter}')

tic = time.time()

print(f'Starting triangulation...', end="", flush=True)

# geom = pg.built_in.Geometry()

geom = pg.opencascade.Geometry(
  characteristic_length_min=0.001,
  characteristic_length_max=0.3,
  )

# Define a simple 2D geometry, a unit square [0,1]x[0,1]

# Corner points
p0 = geom.add_point([0.0, 0.0, 0.0], lcar=mesh_parameter)
p1 = geom.add_point([1.0, 0.0, 0.0], lcar=mesh_parameter)
p2 = geom.add_point([1.0, 1.0, 0.0], lcar=mesh_parameter)
p3 = geom.add_point([0.0, 1.0, 0.0], lcar=mesh_parameter)

# p4 = geom.add_point([0.1, 0.1, 0.0], lcar=mesh_parameter)
#p5 = geom.add_point([0.9, 0.9, 0.0], lcar=mesh_parameter)

circ1 = geom.add_disk([0.5, 0.5, 0.0], radius0=0.2, char_length=0.01)
circ2 = geom.add_disk([0.5, 0.5, 0.0], radius0=0.36, char_length=0.01)
circ3 = geom.add_disk([0.5, 0.5, 0.0], radius0=0.05, char_length=mesh_parameter)

# geom.add_physical(p4)
# p4 = geom.add_point([0.5, 0.8, 0.0], lcar=mesh_parameter)

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

plane = geom.boolean_union([plane_surf, circ1, circ2, circ3])

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


print("Integrating over one triangle")
for i in range(10000):
    kin00 = quadpy.triangle.integrate( lambda x: FEM_hat_grad(triangle_coords[1], 0, 0, x),
                                                    triangle_coords[1], quadpy.triangle.Strang(9) )

print("Integrating over a list of triangles")
for i in range(10):
    kin00 = quadpy.triangle.integrate( lambda x: FEM_hat_grad(triangle_coords[1], 0, 0, x),
                                                    triangle_coords, quadpy.triangle.Strang(9) )