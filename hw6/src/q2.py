# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
import utils 
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
   


    print('********** q2b: Calculation + Visualization ***********')

    # METHOD 1: CLOSED FORM 

    # I = USV^H
    # I = LB = (LQ^-1)(QB) where Q is any 3x3 matrix

    U, S, VH = np.linalg.svd(I,full_matrices=False)
    # print('Orig U', U.shape)
    # print('Orig VH', VH.shape)
    U_new = U[:,:3] # 7, 3
    VH_new = VH[:3,:] # 3, 159039

    S_new  = np.eye(3,3) # 3, 3
    np.fill_diagonal(S_new, S[:3])
    # print('S_new shape', S_new.shape)
    # print('U_new', U_new.shape)
    # print('VH_new', VH_new.shape)
    I_new = U_new @ S_new @ VH_new # use I_new to construct factorization
    
    L = U_new @ np.sqrt(S_new)
    B = np.sqrt(S_new) @ VH_new
    # print('B after shape', B.shape)

    return L, B


    




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
    I, L0, s = loadData()

    print('*********  2a) SVD ***********')

    # # reduce to rank 3, set all singular values to 0 except first 3
    # U, S, VH = np.linalg.svd(I,full_matrices=False)
    # print(U.shape, S.shape, VH.shape)
    # S[3:] = 0
    # S_new  = np.eye(7,7)
    # np.fill_diagonal(S_new, S)
    # I_new = U @ S_new @ VH # use I_new to construct factorization
    # print('I_new', I.shape)

    # # I_new = LB = (LQ^-1)(QB)

    # # input I_new: M x N = 7, 159039
    # # U (7, 7) S (7,) V^H (7, 159039)
    
    # print('U shape', U.shape) # 7x7 (L)
    # # S_new: 
    # print('B shape', VH.shape) # 7x159039  (B)
    # exit(-1)


    print('*********  2b) Estimate Pseudonormals ***********')


    L, B = estimatePseudonormalsUncalibrated(I)
    
    # enforce integrability on pseudonormals
    B = utils.enforceIntegrability(B, s)

    print('B', B.shape)
    print('L', L.shape)
    print('L0', L0.shape)

    print('********** q2c): Estimate Albedos Normals **********')
    albedos, normals_scaled, normals = estimateAlbedosNormals(B)


    print('********** q2b): Display Albedos Normals **********')
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals_scaled, s)

    print('********** q2c): Compare to ground truth lighting **********')
    print('********** q2d): Reconstructing the shape, attempt 1**********')

    surface = estimateShape(normals, s)
   
    plotSurface(surface)

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