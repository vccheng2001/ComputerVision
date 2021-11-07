"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here






import numpy as np
import cv2
from scipy.sparse.linalg import svds, eigs
from helper import *
from util import *
import scipy 
from scipy.ndimage import gaussian_filter
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    Given set of matched image points pts1, pts2, find fundamental matrix 
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
    T = np.eye(3)
    # don't scale last coord (homog!!)
    np.fill_diagonal(T, [1/M, 1/M, 1])

    pts1_norm, pts2_norm = pts1 / M, pts2 / M

    # A matrix = UDV^T
    # Af = 0, solve SVD for 9-element column vector f
    assert N > 8 #(overdetermined)
    A = np.zeros((N,9))
    for i in range(N):
        # http://cs.brown.edu/courses/cs143/proj5/
        x, xp = pts1_norm[i][0], pts2_norm[i][0] # one point
        y, yp = pts1_norm[i][1], pts2_norm[i][1] 
       
        # each pair of corresp points --> one equation 
        A[i] = [x*xp,x*yp,x,y*xp, y*yp, y, xp, yp, 1]
    # http://16720.courses.cs.cmu.edu/lec/two-view_lec15.pdf


    # SVD on matrix A to get f
    # U:(110,110), VH=(9,9)
    U, S, VH = np.linalg.svd(A, full_matrices=True)
    f = VH[-1,:].reshape(3,3)
    F = refineF(f, pts1_norm, pts2_norm)


    # unscale
    F_unnorm = T.T @ F @ T
    

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
    # C1: 3x4
    # C2: 3x4 
    # return: w = [x y z 1]

    err = 0 
    N, _ = pts1.shape # N x 3
    

    # least-squares triangulation
    '''
    for each point i, solve for 3D coordinates 
    wi = [xi, yi, zi] such that when they 
    are projected back to two images, they are
    close to original 2D points. 


    '''    

    # (3x4)@(4x1)    pts1: (3x1) homog
    # C1 @ wi_homog should be lambda*pts1
    # C2 @ wi_homog should be lambda*pts2

    # for each corresp point pair, A is a 4x4 matrix 

    # for each point i, wi is a (4x1) homog. world coord
    w = np.zeros((N,3))
    for i in range(N):

        # pts1 -> C1
        # pts2 -> C2 

        xi, yi = pts1[i]
        xpi, ypi = pts2[i] 
        

        # for each point i,
        # Ai is 4x4 matrix, can solve wi up to scale

        Ai = np.array([xi*C1[2] - C1[0],
                        yi*C1[2] - C1[1],
                        xpi*C2[2] - C2[0],
                        ypi*C2[2] - C2[1]])
        
        # Method 1: eigenvec assoc w/smallest eigenval of Ai^T @ Ai 
        eigenvals, eigenvecs = np.linalg.eig(Ai.T@Ai)
        min_eigenval_idx = np.argmin(eigenvals)
        # NOTE: stores in ith column of np.linalg.eig, not row! 
        # (this is why transpose is necessary)
        # 4 x 1
        wi = eigenvecs.T[min_eigenval_idx]
        # make homogeneous (scale == 1) by dividing by scale 
        wi = (wi / wi[-1])
        w[i] = wi[:3]


        # Method 2: SVD
        # U, S, VH = np.linalg.svd(Ai, full_matrices=True)
        # wi = VH[-1,:]
        # # Divide by scale
        # wi = (wi / wi[-1])
        # w[i] = wi[:3]

        # project 3D back to 2D image points (p)
        # (Cam intrinsics @ Rot/Trans to World) @ World point = projected image point
        # this is 2D homogeneous 
        pts1i_hat = C1 @ wi # C1 @ wi = (K1M1)@wi = (K1@[I|0]) @ wi
        pts2i_hat = C2 @ wi # C2 @ wi = (K2M2)@wi = (K2@[R|t]) @ wi

        # make nonhomog
        pts1i_hat = (pts1i_hat / pts1i_hat[-1])[:2]
        pts2i_hat = (pts2i_hat / pts2i_hat[-1])[:2]



        # compare with original pts1[i], pts2[i]
        # sum over all points 
        err_i = np.linalg.norm(pts1[i]-pts1i_hat)**2 + np.linalg.norm(pts2[i]-pts2i_hat)**2
        err += err_i


    # w: (N, 3)
    return w, err


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
def epipolarCorrespondence(im1, im2, F, x1, y1, wsize=4):

    '''
    given (x1,y1) pixel in im1
    return pixel corresponding to (x1,y1) in im2

    use F to search over set of pixels lying along epipolar line
    given F,x: find xp
    '''
    # get window around pixel (x1, y1)
    win1 = getWindow(im1, (x1,y1), wsize=wsize)
   
    x = np.array([x1,y1,1])
    lp = F @ x # pt2 must lie on this epipolar line

    # points along lp
    pts = getPointsAlongLine(im2, lp)
    
    # best distance, best pt in image 2
    bestDist = np.inf
    bestPt = None

    # for each point along epipolar line l', get dist 
    pts = pts[1:-1, :]
    for pt in pts:
        if pt[0] == 0 or pt[1] == 0: continue
        win2 = getWindow(im2, pt,  wsize=wsize)
        
        dist = computeWindowDist(win1, win2)
        
        if dist < bestDist:
            bestDist = dist
            bestPt = pt

    # print('bestDist', bestDist)
    # print('bestPt', bestPt)
    return bestPt # (x2, y2)

        
    


def getWindow(im, pixel, wsize):
    cx, cy = pixel
    cx, cy = int(cx), int(cy)

    window = im[cy-(wsize//2):cy+(wsize//2), cx-(wsize//2):cx+(wsize//2)]
    return window
'''
Computes distance between two windows' brightness
'''
def computeWindowDist(win1, win2):
    # flatten into 1D vector 
    win1 = win1.flatten()
    win2 = win2.flatten()

    # Euclidean dist 
    dist = np.linalg.norm(win1 - win2, ord=1)
    return dist 


'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter
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
