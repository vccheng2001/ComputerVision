import numpy as np
from numpy.linalg.linalg import _matrix_rank_dispatcher
import cv2
import itertools

def computeH(x1, x2):
    print('x1', x1.shape, 'x2', x2.shape)
    N, _ = x1.shape # N x 2
    
    # fill in 
    row = 0
    A = np.zeros((2*N, 9))
    for i in range(0, len(x2)):
        x, xp  = x1[i][0], x2[i][0]
        y, yp = x1[i][1], x2[i][1]

        A[row] = [x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp]
        A[row+1] = [0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp]

        row += 2

    # Singular Value Decomposition 
    U, S, VH = np.linalg.svd(A, full_matrices=True)

    # last col of V: last row of VH
    H2to1 = VH[8, :]
    # reshape to become transformation/homography matrix
    H2to1 = H2to1.reshape(3,3)
    print('H2to1', H2to1)
    return H2to1


# x1 = np.random.rand(4,2)
# x2 = np.random.rand(4,2)
# computeH(x1,x2)

def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points

    x1_centroid = np.mean(x1, axis=0)
    x2_centroid = np.mean(x2, axis=0)

    # Normalize: Shift the origin of the points to the centroid
    x1 -= x1_centroid
    x2 -= x2_centroid

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_dist_x1 = np.max(np.linalg.norm(x1-x1_centroid))
    max_dist_x2 = np.max(np.linalg.norm(x2-x2_centroid))

    print('max dists', max_dist_x1, max_dist_x2)

    norm_factor_x1  = max_dist_x1 / np.sqrt(2)
    norm_factor_x2  = max_dist_x2 / np.sqrt(2)

    x1_norm = x1 / norm_factor_x1
    x2_norm = x2 / norm_factor_x2

    #Similarity transform 1


    #Similarity transform 2


    #Compute homography x2_tilde to x1_tilde
    H2to1 = computeH(x1_norm, x2_norm)

    print('H in computeH_norm', H2to1)
    #Denormalization
    

    return H2to1




def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    bestH = None
    bestInliers = None
    bestNumInliers = -np.inf

    for i in max_iters:

        H = computeH(locs1, locs2)

        pts = H @ locs2 # apply transform to each pt in locs2

        inliers = pts < inlier_tol

        numInliers = np.count_nonzero(inliers)
        if numInliers > bestNumInliers:
            bestNumInliers = numInliers
            bestInliers = inliers
            bestH = H

    return bestH, bestInliers
        



    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    

    #Create mask of same size as template

    #Warp mask by appropriate homography

    #Warp template by appropriate homography

    #Use mask to combine the warped template and the image
    
    return composite_img


