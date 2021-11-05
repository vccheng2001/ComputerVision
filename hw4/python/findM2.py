'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import numpy as np
import matplotlib.pyplot as plt
import submission as sub
from helper import * 

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

# projective camera matrices M1, M2 
# given M1 is fixed at [I, 0]
# M2 can be retrieved up to a scale, 4-fold rotation ambiguityt
# M2 = [R | t]
pts1, pts2 = data['pts1'], data['pts2']

# test points
p1, p2 = data['pts1'][0], data['pts2'][0]
p1 = sub.make_homogeneous(p1)
p2 = sub.make_homogeneous(p2)

N = data['pts1'].shape[0]
M = 640

# 2.1
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'

data = np.load('../data/intrinsics.npz')
K1, K2 = data['K1'], data['K2']
# calculate essential matrix
E = sub.essentialMatrix(F8, K1, K2)


# projective camera matrices M1, M2 
# M1 = [I | 0], M2 = [R | t]
M1 = np.hstack((np.eye(3), np.zeros((3,1))))
possible_M2 = camera2(E) 

# find the correct M2
# test four solutions through triangulations 
# returns (Nx3) world coordinates 
best_M2 = None
best_err = np.inf
C2 = None 
best_w = None
for i in range(4):
    # candidate M2 
    M2 = possible_M2[i]
    # mult given intrinsics matrices with 
    # solution for canonical camera matrices 
    # to get final camera matrices 
    # get camera matrices 
    C2 = K2 @ M2 
    C1 = K1 @ M1 
    w, err = sub.triangulate(C1, pts1, C2, pts2)
    if err < best_err:
        best_err = err 
        best_M2 = M2 
        best_w = w
        best_C2 = C2 

# Print saving best M2
np.savez('q3_3.npz', best_M2, best_C2, best_w)

