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
import plotly.graph_objects as go

data = np.load('../data/templeCoords.npz')
data_some_corresp = np.load('../data/some_corresp.npz')

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

# 288 selected points in im1
x1, y1 = data['x1'], data['y1']
pts1 = np.hstack((x1,y1))
N, _ = pts1.shape
M = max(im1.shape)

# Get Fundamental Matrix given random set of image correspondences 
F8 = sub.eightpoint(data_some_corresp['pts1'], data_some_corresp['pts2'], M)

# Camera Intrinsics 
intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']

# Essential matrix from Fundamental Matrix, Camera Intrinsics
E = sub.essentialMatrix(F8, K1, K2)

# Epipolar correspondences 
x1s, y1s, x2s, y2s = [],[], [], []
for x1,y1 in pts1:
    x2, y2 = sub.epipolarCorrespondence(im1, im2, F8, x1, y1, wsize=8)
    x2s.append(x2)
    y2s.append(y2)
    x1s.append(x1)
    y1s.append(y1)

# epipolarMatchGUI(im1, im2, F8)

# matching points in im2
pts2 = np.stack((y2s, x2s), axis=1) # 288, 2
pts1 = np.stack((y1s, x1s), axis=1) # 288, 2

# Find best possible rot/trans matrix M2 from essential matrix 
M1 = np.hstack((np.eye(3), np.zeros((3,1))))
possible_M2 = camera2(E)
bestM2, bestw = getBestM2(M1, possible_M2, pts1, pts2, K1, K2)

# Final camera matrix: intrinsic + extrinsic (R/T)
C2 = K2 @ bestM2 
C1 = K1 @ M1 

# Triangulate to get world coordinates 
P, err = sub.triangulate(C1, pts1, C2, pts2)
print('err', err)

# Plot 3D coordinates
fig = go.Figure(data=
                [go.Scatter3d(x=P[:,0], y=P[:,1], z=P[:,2],
                mode='markers',
                marker=dict(
                    size=3),
                )])
fig.show()
print("*******SAVING 4.2*********")
np.savez('q4_2.npz', F=F8, M1=M1, M2=bestM2, C1=C1, C2=C2)

