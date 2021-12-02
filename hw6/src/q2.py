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

    
    U, S, VH = np.linalg.svd(I,full_matrices=False)

    U_new = U[:,:3] # 7, 3
    VH_new = VH[:3,:] # 3, 159039

    # Rank 3
    S_new  = np.eye(3,3) 
    np.fill_diagonal(S_new, S[:3])
 
    I_new = U_new @ S_new @ VH_new 
    
    # I=LB=LQ(Q^-1)B
    # I = USV^T = Usqrt(S)sqrt(S)V^T
    L = U_new @ np.sqrt(S_new)
    B = np.sqrt(S_new) @ VH_new

    return L, B



if __name__ == "__main__":
    I, L0, s = loadData()

    print('*********  2b) Estimate Pseudonormals ***********')


    L, B = estimatePseudonormalsUncalibrated(I)

    print('Comparing light matrices')
    print('L0', L)
    print('L', L)
    
    # enforce integrability on pseudonormals
    B = utils.enforceIntegrability(B, s)

    # # Bas-relief ambiguity
    # mu = 0.5
    # v = 0.2
    # lam = 0.5
    # G = np.array([[1,0,0],[0,1,0],[mu,v,lam]])

    # B = np.linalg.inv(G).T @ B

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
