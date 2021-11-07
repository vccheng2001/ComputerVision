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

# data = np.load('../data/some_corresp.npz')
# im1 = plt.imread('../data/im1.png')
# im2 = plt.imread('../data/im2.png')
# intrinsics = np.load('../data/intrinsics.npz')

# # projective camera matrices M1, M2 
# # given M1 is fixed at [I, 0]
# # M2 can be retrieved up to a scale, 4-fold rotation ambiguity
# # M2 = [R | t]
# pts1, pts2 = data['pts1'], data['pts2']

# N = data['pts1'].shape[0]
# M = 640

# # # 2.1
# F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
# assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'

# # # intrinsic camera parameter matrix 
# K1, K2 = intrinsics['K1'], intrinsics['K2']
# # calculate essential matrix from fund matric, intrinsic camera param matrix
# E = sub.essentialMatrix(F8, K1, K2)


# # projective camera matrices M1, M2 
# # M1 = [I | 0], M2 = [R | t]
# M1 = np.hstack((np.eye(3), np.zeros((3,1))))
# possible_M2 = camera2(E) 


def getBestM2(M1, possible_M2, pts1, pts2, K1, K2):
    # find the correct M2
    # test four solutions through triangulations 
    # returns (Nx3) world coordinates 
    best_M2 = None
    best_C1 = None
    best_C2 = None
    best_err = np.inf
    best_w = None
    for i in range(4):
        # print(f'Trying M2 #{i}.......')
        # candidate M2 
        M2 = possible_M2[:,:,i]
        # print('M', M1.shape, M2.shape) # 4 x 4 
        # mult given intrinsics matrices with 
        # solution for canonical camera matrices 
        # to get final camera matrices 
        # get camera matrices 
        C2 = K2 @ M2 
        C1 = K1 @ M1 
        w, err = sub.triangulate(C1, pts1, C2, pts2)
        # print(f'Got error: {err}')
        if err < best_err:
            best_err = err 
            best_M2 = M2 
            best_C2 = C2
            best_C1 = C1
            best_w = w
    return best_M2, best_w 

# p1, p2 = pts1[0], pts2[0]
# print(f'image points: {p1}, {p2}')
# print(f'projected 3d: {best_w[0]}')


# print(f'Best err: {best_err}, best_M2: {best_M2}')
# # Print saving best M2
# print("**** SAVING 3.3 ******")
# np.savez('q3_3.npz', M2=best_M2, C2=best_C2, P=best_w)