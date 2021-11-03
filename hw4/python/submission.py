"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here






import numpy as np
import cv2
from scipy.sparse.linalg import svds, eigs
from helper import *

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    

    '''
    @input: point correspondences pts1 = (u,v), pts2=(up,vp)
    each point generates one constraint on fundamental matrix F
    and must satisfy epipolar constraint equation

    x'^TFx = 0
    '''
    N, _ = pts1.shape
    # scale by dividing by max of image's width, height
    # if x_norm = Tx, F_unnorm = T^T @ F @ T 


    T = np.eye(3)
    np.fill_diagonal(T, 1/M) # scaling matrix
    pts1_norm, pts2_norm = pts1, pts2# pts1 / M, pts2 / M#T @ pts1, T @ pts2

    # A matrix = UDV^T
    # Af = 0, solve SVD for 9-element column vector f
    assert N > 8 #(overdetermined)
    A = np.zeros((N,9))
    for i in range(N):
        # http://cs.brown.edu/courses/cs143/proj5/
        x, xp = pts2_norm[i][0], pts1_norm[i][0] # one point
        y, yp = pts2_norm[i][1], pts1_norm[i][1] 
    
        

        # each pair of corresp points --> one equation 
        A[i] = [x*xp,x*yp,x,y*xp, y*yp, y, xp, yp, 1]
    # http://16720.courses.cs.cmu.edu/lec/two-view_lec15.pdf


    # SVD on matrix A to get f
    # U:(110,110), VH=(9,9)
    U, S, VH = np.linalg.svd(A, full_matrices=True)
    # least square solution: singular vector 
    # corresponding to smallest singular value of A 
    # last column of V == last row of VH
    

    f = VH[8,:]
    # Divide by scale
    f = f / f[-1]

    # A: (N x 9), f: (9 x 1)
    print('A @ f should be 0, is', A @ f)
    
    # 3x3 fundamental matrix F
    F = f.reshape(3,3)

    # enforce singularity condition of F before unscaling 
    # make sure F is rank 2

    # set sigma3 = 0
    Uf, Sf, VfH = np.linalg.svd(F, full_matrices=True)
    Sf[2] = 0
    Sf_sing_vec = Sf

    # create new diag matrix where smallest singular value == 0
    Sf_sing_matrix = np.eye(3)
    np.fill_diagonal(Sf_sing_matrix, Sf_sing_vec)
    print('Sf_sing_matrix', Sf_sing_matrix)

    # get rank-2 fundamental matrix
    F = Uf @ Sf_sing_matrix @ VfH

    # unscale
    print('F', F)
    F_unnorm = T.T @ F @ T

    # check: 
    # print('pts1[0]', pts1[0])
    p1 = make_homogeneous(pts1[0])
    p2 = make_homogeneous(pts2[0])
    print('xpi^T @ F @ xi should be 0, is', (p2).T @ F_unnorm @ (p1))
    return F_unnorm
    

def make_homogeneous(pt):
    return np.concatenate((pt, np.ones(1)), axis=None)


    '''
    Fundamental matrix F: 3x3 matrix
    -relates corresponding points in stereo images.
    In epipolar geometry: corresponding points x and x'
    - Fx describes line on which x' on other image must lie
    - x'TFx = 0
    - rank 2, up to scale
    - can be estimated given at least seven point correspondences
    '''

    # Compute fundamental matrix from 8 point alg
    # if images taken by same camera, K ==K'
    E = Kp.T @ F @ K

    assert xp.T @ F @ x == 0
    # from E, get (P0,P0p), (P1,P1p), ....


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    #E = Kp.T @ F @ K

    
    return K2.T @ F @ K1


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''


''' Planar triangulation
- collection of N 2D points vertex=(N,2)
- collection of triangles faces (M, 2) 

Building triangulation of convex hull of points
Compute Delaunay triangulation of points
https://www.numerical-tours.com/matlab/meshproc_1_basics_2d/
Iterative filtering 


Computer Vision

Structure: scene geometry
Motion: camera geometry

Reconstruction: estimate both, 2D to 2D corresp
Triangulation: know motion, 2D to 2D
Pose Estimation: know structure, 3D to 2D 

http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
- Given set of noisy matched points {xi, x'i}
- Given camera matrices C1, C2
- estimate 3D point X
- x = PX (can we compute X from single corresp x?)
- since meas are noisy, there's no point satisfying both x'=P'X and x=PX
-   need to find line of best fit
- since homogeneous coords, solve up to a scale (x=alphaPX)
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation


    pass



'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
