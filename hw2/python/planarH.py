import numpy as np
from numpy.linalg.linalg import _matrix_rank_dispatcher
import cv2
import itertools
np.set_printoptions(precision=3, suppress=True) 
# x1, x2: N x 2


def computeH(x1, x2):


    # print('x1', x1)
    # print('x2', x2)

    # NOTE: SHOULD be X1 = HX2 
    H_true, mask = cv2.findHomography(x2, x1, cv2.RANSAC, 5.0)
    # print('H_true', H_true)

    N, _ = x1.shape # N: number of point pairs
    
    # fill in 
    row = 0
    A = np.zeros((2*N, 9))
    for i in range(N):
        x, xp  = x2[i][0], x1[i][0]
        y, yp = x2[i][1], x1[i][1]

        A[row] = [x, y, 1, 0, 0, 0, -xp*x,-xp*y,-xp]
        A[row+1] = [0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp]
       
        row += 2
    
    # Solve for h such that Ah = 0

    # Method 1: Eigenvector corresponding to minimum eigenvalue
    eigenvals, eigenvecs = np.linalg.eig(A.T@A)

    min_eigenval_idx = np.argmin(eigenvals)
    min_eigenval = eigenvals[min_eigenval_idx]

    # NOTE: stores in ith column of np.linalg.eig, not row! (this is why transpose is necessary)
    H_eig = eigenvecs.T[min_eigenval_idx]

    # Method 2: Singular Value Decomposition 
    U, S, VH = np.linalg.svd(A, full_matrices=True)
    H_svd = VH[8,:]


    # divide by scale
    H_svd = (H_svd / H_svd[-1]).reshape(3,3)
    H_eig = (H_eig / H_eig[-1]).reshape(3,3)

    # print("H_svd", H_svd)
    # print("H_eig", H_eig)


    # assert np.allclose(H_svd, H_eig)
    # assert np.allclose(H_svd, H_true)

    # reshape to become transformation/homography matrix
    
    return H_svd

# np.random.seed(10)

# x1 = np.array([[93,-7],[293,3],[1207,7],[1218,3]])
# x2 = np.array([[63,0],[868,-6],[998,-4],[309,2]])

# H = computeH(x1,x2)

# TEST ALL 
# N, _ = x2.shape
# x2_inp = np.hstack((x2, np.ones((N, 1)))).T
# print(x2_inp.shape)
# res = H @ x2_inp
# print('res', res / res[-1]) # make sure this is x1

# TEST ONE
# x2_inp = np.array([868,-6,1])
# res = H @ x2_inp
# print('H', H)
# print('res', res/res[-1]) # should be [293,3, 1]
# exit(-1)

def computeH_norm(x1, x2): # N x 2
    N1, _ = x1.shape
    N2, _ = x2.shape

    #Q2.2.2
    #Compute the centroid of the points
    print('x1', x1)
    print('x2', x2)

    x1 = np.hstack((x1.astype(np.float64), np.ones((N1, 1))))
    x2 = np.hstack((x2.astype(np.float64), np.ones((N2, 1))))

    x1_centroid = np.mean(x1, axis=0).astype(np.float64)
    x2_centroid = np.mean(x2, axis=0).astype(np.float64)


    print('x1x2centroid', x1_centroid, x2_centroid)
    # 1. translate the origin of the points to the centroid
    x1_norm = x1 - x1_centroid
    x2_norm = x2 - x2_centroid
    print('x1norm', x1_norm)
    max_dist_x1 = np.max(np.linalg.norm(x1_norm-x1_centroid), axis=0)
    max_dist_x2 = np.max(np.linalg.norm(x2_norm-x2_centroid), axis=0)

    # 2. scale down by norm_factor (make max dist sqrt(2))
    norm_factor_x1  = max_dist_x1 / np.sqrt(2)
    norm_factor_x2  = max_dist_x2 / np.sqrt(2)
    x1_norm = x1_norm / norm_factor_x1
    x2_norm = x2_norm / norm_factor_x2
    # print('NO T x1norm', x1_norm)
    # print('NO T x2norm', x2_norm)
    
    # Compute homography x2_tilde to x1_tilde
    x1_norm[:,2] = 1
    x2_norm[:,2] = 1
    # print('NO T x1norm', x1_norm)
    # print('NO T x2norm', x2_norm)
    H_norm = computeH(x1_norm, x2_norm)

    # print('H_norm', H_norm)
    # print('x1n', x1_norm.shape, x1_norm) # N x 3
    # print('x2n', x2_norm.shape, x2_norm) # N x 3

    # r = H_norm.T @ np.array([-0.703,0.005,1])
    # print('r', r/r[-1])
    #Similarity transform 1

    trans_1 = np.array([[1,0,-x1_centroid[0]],
              [0,1,-x1_centroid[1]],
               [0,0,1]])
    scale_1 = np.array([[1/norm_factor_x1,0,0],
              [0,1/norm_factor_x1,0],
               [0,0,1]])


    #Similarity transform 2
    trans_2 = np.array([[1,0,-x2_centroid[0]],
              [0,1,-x2_centroid[1]],
               [0,0,1]])
    scale_2 = np.array([[1/norm_factor_x2,0,0],
              [0,1/norm_factor_x2,0],
               [0,0,1]])
            
    T1 = scale_1 @ trans_1  
    T2 = scale_2 @ trans_2 
    # print('T1', T1)
    # print('T2', T2)
    x1pnorm = T1 @ x1.T # (3x3) @ (3x4)
    x2pnorm = T2 @ x2.T
    # print('WITH T x1norm', x1pnorm.T)
    # print('WITH T x2norm', x2pnorm.T)

    # Ensure 
    # x1_tilde = T1 @ x1

    

    # Return homography of unnormalized coordinates

    # print('inv', np.linalg.inv(T1))
    # Denormalization
    H = np.linalg.inv(T1) @ H_norm @ T2
    print('a', H @ (T2 @ x2.T))
    print('b', T1 @ x1.T)
    return H



# # # TEST ONE
# x1 = np.random.rand(4,2)
# x2 = np.random.rand(4,2)
# # x1 = np.array([[93,-7],[293,3],[1207,7],[1218,3]])
# # x2 = np.array([[63,0],[868,-6],[998,-4],[309,2]])

# H = computeH_norm(x1,x2)
# print('H', H)
# x2_inp = np.array([309,2,1])
# res = H @ x2_inp
# print('res', res/res[-1]) # should be [293,3, 1]

# exit(-1)

def computeH_ransac(locs1, locs2, opts):


    N = min(locs1.shape[0], locs2.shape[0])

    locs1 = locs1[:N,:]
    locs2 = locs2[:N,:]

    print('N', N)
    
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    bestH = None
    bestInliers = None
    bestNumInliers = 0

    for i in range(max_iters):
        sample_size = 4

        # choose random subset
        rand_idx = np.random.choice(N, size=sample_size, replace=False)

        subset1 = locs1[rand_idx] 
        subset2 = locs2[rand_idx]


        H = computeH_norm(subset1, subset2)

        # H converts point in locs2 to point in locs1
        col = np.ones((N,1))
        pts_to_transform = np.hstack((locs2, col)).T # 3 x N

        # apply transform to each pt in locs2 to get pts in locs1
        transformed_pts = H @ pts_to_transform # 3 x 3 @ 3 x N = 3 @ N
        transformed_pts = transformed_pts / transformed_pts[-1]
        transformed_pts = (transformed_pts[:2,:]).T # N x 2

        dist = np.linalg.norm(locs1 - transformed_pts, axis=1)
        idx = np.argwhere(dist < inlier_tol).flatten() # inlier_tol = 2
        numInliers = len(idx)

        if numInliers == 0:
            print('No inliers found')
        else:
            print(f'{numInliers} inliers found')
            inliers = transformed_pts[idx]
    
        # repeat each iteration with current LARGEST set of inliers
        if numInliers > bestNumInliers:
            bestNumInliers = numInliers
            bestInliers = inliers
            bestH = H
    return bestH, bestInliers
        


# x1 = np.array([[93,-7],[293,3],[1207,7],[1218,3]])
# x2 = np.array([[63,0],[868,-6],[998,-4],[309,2]])

# class Opts:
#     def __init__(self):
#         self.inlier_tol = 2
#         self.max_iters = 500
# opts = Opts()
# bestH, bestInliers = computeH_ransac(x1,x2, opts)

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


