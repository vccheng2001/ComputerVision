import numpy as np
from numpy.linalg.linalg import _matrix_rank_dispatcher
import cv2
import itertools
np.set_printoptions(precision=3, suppress=True) 
# x1, x2: N x 2
def computeH(x1, x2):


    # SWAP
    x1[:, [1, 0]] = x1[:, [0, 1]]
    x2[:, [1, 0]] = x2[:, [0, 1]]

    H_true, mask = cv2.findHomography(x1, x2, cv2.RANSAC, 5.0)
    print('H_true', H_true)

    N, _ = x1.shape # N: number of point pairs
    
    # fill in 
    row = 0
    A = np.zeros((2*N, 9))
    for i in range(N):
        x, xp  = x1[i][0], x2[i][0]
        y, yp = x1[i][1], x2[i][1]

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

    print("H_svd", H_svd)
    print("H_eig", H_eig)


    assert np.allclose(H_svd, H_eig)
    assert np.allclose(H_svd, H_true)

    # reshape to become transformation/homography matrix
    
    return H_svd

# np.random.seed(10)

x1 = np.array([[93,-7],[293,3],[1207,7],[1218,3]])
x2 = np.array([[63,0],[868,-6],[998,-4],[309,2]])

computeH(x1,x2)

def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points

    x1 = x1.astype(np.float64)
    x2 = x2.astype(np.float64)

    x1_centroid = np.mean(x1, axis=0).astype(np.float64)
    x2_centroid = np.mean(x2, axis=0).astype(np.float64)

    # Normalize: Shift the origin of the points to the centroid
    x1 -= x1_centroid
    x2 -= x2_centroid

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_dist_x1 = np.max(np.linalg.norm(x1-x1_centroid))
    max_dist_x2 = np.max(np.linalg.norm(x2-x2_centroid))


    norm_factor_x1  = max_dist_x1 / np.sqrt(2)
    norm_factor_x2  = max_dist_x2 / np.sqrt(2)

    x1_norm = x1 / norm_factor_x1
    x2_norm = x2 / norm_factor_x2


    # print('x1_norm',x1_norm)
    # print('x2_norm', x2_norm)
    # T converts x to x_tilde
    # H converts x2 to x1 
    #Similarity transform 1


    #Similarity transform 2


    #Compute homography x2_tilde to x1_tilde
    H_norm = computeH(x1_norm, x2_norm)
    # H = np.inv(T1) @ H_norm @ T2

    # print('H in computeH_norm', H2to1)
    #Denormalization
    
    H = H_norm

    return H




def computeH_ransac(locs1, locs2, opts):

    N1, _ = locs1.shape
    N2, _ = locs2.shape
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    bestH = None
    bestInliers = None
    bestNumInliers = 0

    for i in range(max_iters):
        N = 50

        # choose random subset
        rand_idx = np.random.choice(min(N1,N2), size=N, replace=False)

        subset1 = locs1[rand_idx] 
        subset2 = locs2[rand_idx]

        # print('subset1', subset1)
        # print('subset2', subset2)


        H = computeH(subset1, subset2)
        # H converts point in locs2 to point in locs1

        col = np.ones((N,1))
        pts_to_transform = np.hstack((subset2, col))

        # print('pts_to_transf', pts_to_transform.shape)

        # apply transform to each pt in locs2 to get pts in locs1
        # print('pts_to_transform.T', pts_to_transform.T)
        transformed_pts = H @  pts_to_transform.T # 3 x 3 @ 3 x N = 3 @ N
        transformed_pts = (transformed_pts[:2, :]).T # N x 2
        # print('transf', transformed_pts.shape) 
        dist = np.linalg.norm(subset1 - transformed_pts, axis=1)
        idx = np.argwhere(dist < inlier_tol).flatten() # inlier_tol = 2
        numInliers = len(idx)

        if numInliers == 0:
            print('No inliers found')
        else:
            print('numinliers', numInliers)
            print('dist', dist)
            print('IDX', idx)
            inliers = transformed_pts[idx]
    
        # repeat each iteration with current LARGEST set of inliers
        if numInliers > bestNumInliers:
            bestNumInliers = numInliers
            bestInliers = inliers
            bestH = H
    return bestH, bestInliers
        


# x1 = np.array([[40., 96.], [56., 219.], [37., 667.], [56., 720.]])
# x2 = np.array([[40., 96.], [56., 219.], [37., 667.], [56., 720.]])

# class Opts:
#     def __init__(self):
#         self.inlier_tol = 2
#         self.max_iters = 500
# opts = Opts()
# H = computeH_ransac(x1,x2, opts)
# # print('x1', x1)
# # print('x2', x2)
# col = np.ones((x2.shape[0],1))
# print(H @ np.hstack((x2, col)).T)


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


