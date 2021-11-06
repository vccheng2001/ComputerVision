"""
Check the dimensions of function arguments
This is *not* a correctness check
Written by Chen Kong, 2018.
"""
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
from helper import * 

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

# test points
p1, p2 = data['pts1'][0], data['pts2'][0]
p1 = sub.make_homogeneous(p1)
p2 = sub.make_homogeneous(p2)

N = data['pts1'].shape[0]
M = 640





# 2.1
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'
# displayEpipolarF(im1, im2, F8)

np.savez('q2_1.npz', F8)

intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']

# calculate essential matrix
E = sub.essentialMatrix(F8, K1, K2)
print('Essential matrix: ', E)
np.savez('q3_1.npz', E)

print('Checking essential matrix, should be 0', p2.T @ E @ p1)



# 3.1
C1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
C2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)

P, err = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
assert P.shape == (N, 3), 'triangulate returns Nx3 matrix P'
assert np.isscalar(err), 'triangulate returns scalar err'




# 4.1
print('********* EPIPOLAR CORRESPONDENCES *********')
x2, y2 = sub.epipolarCorrespondence(im1, im2, F8, data['pts1'][0, 0], data['pts1'][0, 1])

assert np.isscalar(x2) & np.isscalar(y2), 'epipolarCorrespondence returns x & y coordinates'
np.savez('q4_1.npz', F8, data['pts1'][0, 0], data['pts1'][0, 1])

epipolarMatchGUI(im1, im2, F8)

# 5.1
"""
You can opt to uncomment this if extra credit q5 is implemented. Note this only checks formatting. 
"""
# F, inliers = sub.ransacF(data['pts1'], data['pts2'], M)
# assert F.shape == (3, 3), 'ransacF returns 3x3 matrix'

# # 5.2
# r = np.ones([3, 1])
# R = sub.rodrigues(r)
# assert R.shape == (3, 3), 'rodrigues returns 3x3 matrix'

# R = np.eye(3);
# r = sub.invRodrigues(R)
# assert (r.shape == (3, )) | (r.shape == (3, 1)), 'invRodrigues returns 3x1 vector'

# # 5.3
# K1 = np.random.rand(3, 3)
# K2 = np.random.rand(3, 3)
# M1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
# M2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
# r2 = np.ones(3)
# t2 = np.ones(3)
# x = np.concatenate([P.reshape([-1]), r2, t2])
# residuals = sub.rodriguesResidual(K1, M1, data['pts1'], K2, data['pts1'], x)
# print(residuals.shape)
# print(N)
# assert residuals.shape == (4 * N,), 'rodriguesResidual returns vector of size 4N'

# M2, P = sub.bundleAdjustment(K1, M1, data['pts1'], K2, M2, data['pts1'], P)
# assert M2.shape == (3, 4), 'bundleAdjustment returns 3x4 matrix M'
# assert P.shape == (N, 3), 'bundleAdjustment returns Nx3 matrix P'

# print('Format check passed.')
