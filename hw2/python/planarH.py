import numpy as np
from numpy.linalg.linalg import _matrix_rank_dispatcher
import cv2
import itertools
np.set_printoptions(precision=3, suppress=True) 

def computeH(x1, x2):

    # ground truth H (for verification, compare with H_eig, H_svd)
    H_true, mask = cv2.findHomography(x2, x1, cv2.RANSAC, 5.0)

    N, _ = x1.shape # N: number of point pairs
    
    # A matrix
    row = 0
    A = np.zeros((2*N, 9))
    for i in range(N):
        x, xp  = x2[i][0], x1[i][0]
        y, yp = x2[i][1], x1[i][1]

        A[row] = [x, y, 1, 0, 0, 0, -xp*x,-xp*y,-xp]
        A[row+1] = [0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp]
       
        row += 2
    
    # Solve for h such that Ah = 0
    # *****  Method 1: Eigenvector corresponding to minimum eigenvalue ***** 
    eigenvals, eigenvecs = np.linalg.eig(A.T@A)

    min_eigenval_idx = np.argmin(eigenvals)
    min_eigenval = eigenvals[min_eigenval_idx]

    # NOTE: stores in ith column of np.linalg.eig, not row! 
    # (this is why transpose is necessary)
    H_eig = eigenvecs.T[min_eigenval_idx]




    # ***** Method 2: Singular Value Decomposition ***** 
    U, S, VH = np.linalg.svd(A, full_matrices=True)
    H_svd = VH[8,:]

    # Divide by scale
    H_svd = (H_svd / H_svd[-1]).reshape(3,3)
    H_eig = (H_eig / H_eig[-1]).reshape(3,3)

    return H_svd


# Compute H_norm with normalized coordinates
# Then use similarity transforms to get H
def computeH_norm(x1, x2):
    N1, _ = x1.shape
    N2, _ = x2.shape
    
    # Make homogeneous coordinates
    x1 = np.hstack((x1.astype(np.float64), np.ones((N1, 1))))
    x2 = np.hstack((x2.astype(np.float64), np.ones((N2, 1))))

    # Normalize
    x1_centroid = np.mean(x1, axis=0).astype(np.float64)
    x2_centroid = np.mean(x2, axis=0).astype(np.float64)


    # Centroid should now be origin
    x1_norm = x1 - x1_centroid
    x2_norm = x2 - x2_centroid
    
    # Get dist from origin
    max_dist_x1 = np.max(np.linalg.norm(x1_norm), axis=0)
    max_dist_x2 = np.max(np.linalg.norm(x2_norm), axis=0)

    # Make max dist sqrt(2)
    norm_factor_x1  = max_dist_x1 / np.sqrt(2)
    norm_factor_x2  = max_dist_x2 / np.sqrt(2)
    x1_norm = x1_norm / norm_factor_x1
    x2_norm = x2_norm / norm_factor_x2

    # Make homogeneous 
    x1_norm[:,2] = 1
    x2_norm[:,2] = 1
    
    # Compute H_norm
    H_norm = computeH(x1_norm, x2_norm)

    # Similarity transforms for x1, x2
    T1 = np.array([[1/norm_factor_x1,0,-x1_centroid[0]/norm_factor_x1],
                        [0,1/norm_factor_x1,-x1_centroid[1]/norm_factor_x1],
                        [0,0,1]])
    
    T2 = np.array([[1/norm_factor_x2,0,-x2_centroid[0]/norm_factor_x2],
                        [0,1/norm_factor_x2,-x2_centroid[1]/norm_factor_x2],
                        [0,0,1]])
   

    # Return homography of unnormalized coordinates
    # Denormalization
    H = np.linalg.inv(T1) @ H_norm @ T2
    # Divide by scale
    H /= H[2,2]
    return H



def computeH_ransac(locs1, locs2, opts):
    N, _ = locs1.shape

    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    bestH = None
    bestInliers = None
    bestNumInliers = 0

    for i in range(max_iters):
        sample_size = 4

        # choose random subset of 4 points
        rand_idx = np.random.choice(N, size=sample_size, replace=False)

        subset1 = locs1[rand_idx] 
        subset2 = locs2[rand_idx]

        # Compute H_norm using those four points
        H = computeH_norm(subset1, subset2)

        # H converts point in locs2 to point in locs1
        col = np.ones((N,1))
        pts_to_transform = np.hstack((locs2, col)).T # 3 x N

        # Apply transform to each pt in locs2 to get pts in locs1
        transformed_pts = H @ pts_to_transform # (3 x 3) @ (3 x N) = 3 @ N
        # Divide by scale
        transformed_pts = transformed_pts / transformed_pts[-1]
        transformed_pts = (transformed_pts[:2,:]).T # N x 2

        # Dist from locs1 to transformed points to get inliers
        dist = np.linalg.norm(locs1 - transformed_pts, axis=1)
        idx = np.argwhere(dist < inlier_tol).flatten() # inlier_tol = 2
        numInliers = len(idx)

        if numInliers != 0:
            inliers = transformed_pts[idx]
    
        # Find best H corresponding to maximum number of inliers
        if numInliers > bestNumInliers:
            bestNumInliers = numInliers
            bestInliers = inliers
            bestH = H

    return bestH, bestInliers
        
