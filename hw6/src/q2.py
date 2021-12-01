# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability

def estimatePseudonormalsUncalibrated(I):

    """  
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    # I = LB
    # Measurement matrix I
    # Lights L
    # Pseudonormals B

    L_inv = L.T @ np.linalg.inv(L @ L.T) # 490, 574, 547
    rett = L_inv.T @ I 

    





    # albedos, normals

    w, h = s


    # reshape albedos into image shape
    albedoIm = np.reshape(albedos, (w,h))
    print(albedoIm[0,0],albedoIm[0,1],albedoIm[0,2])
    print('normals', -normals*255)
    normalIm = np.reshape(-255*normals, (w,h,3)).astype(np.uint8)

    fig = plt.figure()
    plt.imshow(normalIm, cmap=cm.rainbow)
    plt.show()


    # albedo
    fig = plt.figure()
    plt.imshow(albedoIm, cmap=cm.gray)
    plt.show()



    B = None
    L = None

    return B, L


if __name__ == "__main__":

    print('*********  2a) SVD ***********')

    I = 
    U, S, VH = np.linalg.svd(I,full_matrices=False)
    print(U.shape, S.shape, VH.shape)
    S[3:] = 0
    S_new  = np.eye(7,7)
    np.fill_diagonal(S_new, S)

    I_new = U @ S_new @ VH
    print('I_new', I.shape)

    print('*********  2b) ***********')

    estimatePseudonormalsUncalibrated(I)


    # Put your main code here
    pass


# a = np.array([[25,50,0,0,0,-5],
#             [50,200,0,0,0,-20],
#             [0,0,1,0,0,0],
#             [0,0,0,0,0,0],
#             [0,0,0,0,0,0],
#             [-5,-20,0,0,0,3]])

# b = np.array([[0,0,0,0,0,0],
#             [0,325,0,0,0,25],
#             [0,0,1,0,0,0],
#             [0,0,0,0,0,0],
#             [0,0,0,0,0,0],
#             [0,25,0,0,0,3]])
# # [215.27987357  11.73012969   1.           0.98999674   0.
# #    0.        ]
# u, s, v = np.linalg.svd(b, full_matrices=True, compute_uv=True, hermitian=False)

# print('singular values', s)

# #  [326.92943258   1.07056742   1.           0.           0.
# #    0.        ]