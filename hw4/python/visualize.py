'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter

    Determine 3D locations of point correspondences with triangulation
    0) Epipolar correspondence using F8 to get pts2
    1) Get F8 with pts1, pts2
    2) Use this to triangulate(C1, pts1, C2, pts2) to get world points p
'''


"""
Check the dimensions of function arguments
This is *not* a correctness check
Written by Chen Kong, 2018.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import submission as sub
from helper import * 
from findM2 import *

data = np.load('../data/templeCoords.npz')
data_some_corresp = np.load('../data/some_corresp.npz')

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

# 288 selected points in im1
x1, y1 = data['x1'], data['y1']
pts1 = np.hstack((x1,y1))

N, _ = pts1.shape
M = max(im1.shape)

# Get Fundamental Matrix given set of image correspondences 
# F8 = sub.eightpoint(pts1, pts2, M)
F8 = sub.eightpoint(data_some_corresp['pts1'], data_some_corresp['pts2'], M)

# Camera Intrinsics 
intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']

# Essential matrix from Fundamental Matrix, Camera Intrinsics
E = sub.essentialMatrix(F8, K1, K2)

x2s, y2s = [],[]
# Epipolar correspondences 
for x1,y1 in pts1:
    x2, y2 = sub.epipolarCorrespondence(im1, im2, F8, x1, y1)
    x2s.append(x2)
    y2s.append(y2)
# matching points in im2
pts2 = np.stack((x2s, y2s), axis=1) # 288, 2
print('pts2', pts2.shape) # 

# Final Camera Projection Matrix 
C1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
C2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)

# Triangulate to get world coordinates 
# P: N,3
P, err = sub.triangulate(C1, pts1, C2, pts2)
# print('world points P', P)
# plot 3D coords 
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(P[:,0], P[:,1], P[:,2])
pyplot.show()



M1 = np.hstack((np.eye(3), np.zeros((3,1))))
# get possible M2 from essential matrix 
possible_M2 = camera2(E)
M2 = getBestM2(M1, possible_M2, pts1, pts2, C1, C2)
np.savez('q4_2.npz', F8, M1, M2, C1, C2)

