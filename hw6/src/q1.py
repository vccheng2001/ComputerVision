# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
from contextlib import contextmanager
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils
import scipy
import scipy.io
from PIL import Image
import matplotlib.cm as cm
import cv2
import skimage
from scipy import sparse
import scipy.sparse.linalg

import skimage
import skimage.color
from skimage.color import rgb2xyz

def scale_zero_to_one(x):
    xmax = np.max(x)
    xmin = np.min(x)
    return (x-xmin) / (xmax - xmin)


def normalize(v):
    norm = np.linalg.norm(v)
    return norm, v / norm 


def renderNDotLSphere(center, rad, light, pxSize, res, plot):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    # resolution
    r = np.arange(0, res[1], 1)
    c = np.arange(0, res[0], 1)

    y, x= np.meshgrid(r,c)

    # translate image origin to actual origin (offset by res/2)
    x_3d = (x-res[0]/2)*pxSize
    y_3d = (y-res[1]/2)*pxSize
    # sphere equation to solve for z
    z_3d = np.sqrt(rad**2 + 0j - x_3d**2 - y_3d**2)

    x_3d[np.real(z_3d) == 0] = 0
    y_3d[np.real(z_3d) == 0] = 0
    z_3d[np.real(z_3d) == 0] = 0
    n = np.array([x_3d, y_3d ,z_3d])

    n = np.transpose(n, (1,2,0))
    # (3840, 2160, 3) x (3,1) = (3840, 2160)
    ndotl = np.dot(n, light)
    ndotl[ndotl < 0] = 0
 
    ndotl = np.real(ndotl).T

    if plot:
        plt.imshow(ndotl.astype('float64'), cmap="gray")
        plt.show()

    return ndotl



def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    im = cv2.imread(f'{path}input_1.tif', cv2.IMREAD_UNCHANGED)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    assert(im.dtype == np.uint16)

    lum = skimage.color.rgb2gray(im)
    
    # im1 = np.uint16(im1)
    # assert(im1.dtype == np.uint16)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2XYZ)
    # lum = im[:,:,1] # extract luminance Y from XYZ

    s = lum.shape
    
    lum = lum.flatten()

    I = lum
    
    # W(431)*H(369) = 159039 PIXELS * 3 CHANNELS = 477117
    # stack remaining imgs
    for i in range(1,7):
        im = cv2.imread(f'{path}input_{i+1}.tif', cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        assert(im.dtype == np.uint16)

        lum = skimage.color.rgb2gray(im)
        # im = np.uint16(im)


        # cv2.imshow(str(i), im)
        # cv2.waitKey(0)
        # lum = im[:,:,1] # Y
        lum = lum.flatten()
        I = np.vstack((I,lum))

    # relative luminance
    # print('I norm lum', I)
    # print('I type before cast', I.dtype) # float64
    # I = I.astype(np.uint16)
    I = I / np.max(I)
    # print(I, 'I after cast', I.dtype)

    L = np.load(f'{path}/sources.npy').T # (3x7)
    print('L', L.shape) # (3,7)
    print('I', I.shape) # (7, 159039)
    print('s', s)       # (431, 369)

    return I, L, s



def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    """
    # m < n: AA^T invertible 

    # L_inv = np.linalg.inv(L.T @ L) @ L.T # 263, 308, 294


    # ****
    L_inv = L.T @ np.linalg.inv(L @ L.T) # 490, 574, 547
    print(L_inv.shape, I.shape)
    rett = L_inv.T @ I 




    # b = np.linalg.inv(L.T@L)@L@I

    # I = Lb
    # minimize: (I-Lb), solve Lb=I
    # L: M,N
    # I: M, or M, K

    # Numpy linalg.lstsq(a, b, rcond='warn') returns 
    # solution to vector x that satisfies a@x=b

    # Ax = b
    # L^Tb = I
    # L^T: 7p*3p, b: flatten to 3p, I: flatten to 7p
    import copy
    L_orig = copy.deepcopy(L)
    I_orig = copy.deepcopy(I)
    _, P = I.shape # 

    # I:  N= 7*NumPixels 
    # I = I.reshape(I, 7*P) 
    # (3x7)@(PxP) = 
    identity = scipy.sparse.eye(P)
    

    # a: 7x3
    # b: p*p
    # M=7, N=3, P=p, Q=p
    # out: 7p, 3p 

    L = sparse.kron(L,identity).T
    I = np.reshape(I, (7*P))
    I = sparse.csr_matrix(I).T
    

    # L: M=7p, N=3p
    # I: M=7p
    # B: N=3p

    # (1113273, 477117)
    print(f'L should be {7*P} by {3*P}', L.shape) # M x N, M=7p, N=3p
    # (1113273, 1)
    print(f'I should be {7*P}', I.shape) # M=7p
    

    ret = sparse.linalg.lsqr(L, I.toarray())
    b = ret[0]
    b = np.reshape(b, (3, -1))
    print(f'b should be 3 x {P}', b.shape) # (3*159039)

    n_tilde = np.linalg.pinv(L_orig.T) @ I_orig
    
    
    return rett




def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    
    _, P = B.shape
    w = 431
    h = 369
    n_tilde = B

    # method 1 
    #**********
    n_tilde = np.reshape(n_tilde, (3, w, h))

    albedos = []
    normals = np.empty((3, w,h,))



    # r = normalize(np.array([1,2,3]))
    # print(np.linalg.norm(r))
    # exit(-1)
    for u in range(w):
        
        for v in range(h):

            norm, unit_vec = normalize(n_tilde[:, u,v])


            albedos.append(norm)
            normals[:,u,v] = unit_vec

    normals_scaled = ((normals + 1)/2)
    normals_scaled = np.reshape(normals_scaled, (3, P))
    albedos = np.array(albedos)

    # scale to between 0,1
    albedos = scale_zero_to_one(albedos)
    
    print('max min albedos', np.max(albedos), np.min(albedos))
    print('max min normals', np.max(normals), np.min(normals))


    # scale to between -1,1
    # normals = 2*scale_zero_to_one(normals)-1
    return albedos, normals_scaled, normals
    # n_tilde = B # 3x1 col vec PER PIXEL 
  
    

    # method 2 (same)
    #**********
    # albedos = np.empty(P)
    # normals = np.empty((3,P))

    # # albedo_i = mag(n_tilde_i)
    # # normal_i = n_tilde_i / albedo_i
    # for i in range(P):
    #     magnitude = np.linalg.norm(n_tilde[:,i]) # magnitude
    #     albedo = magnitude # albedo
    #     normal = n_tilde[:,i] / magnitude
    #     albedos[i] = albedo # vector of length <pixels>
    #     normals[:, i] = normal # s x 3

    # # (num_pixels), (3 x num_pixels)
    # return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    w, h = s

    # scale normals from [-1,1] range to [0,1] range

    # normals = ((normals + 1)/2)#*255

    print('normals should be [0,1]', np.max(normals), np.min(normals) )

    # reshape albedos into image shape
    albedoIm = np.reshape(albedos, (w,h))
    print(albedoIm[0,0],albedoIm[0,1],albedoIm[0,2])
    normalIm = np.reshape(normals, (w,h,3))#.astype(np.uint8)

    fig = plt.figure()
    plt.imshow(normalIm, cmap=cm.rainbow)
    plt.show()


    # albedo
    fig = plt.figure()
    plt.imshow(albedoIm, cmap=cm.gray)
    plt.show()

    return albedoIm, normalIm



def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    

    # normals = utils.enforceIntegrability(normals, s)
    # print('normals after enforcing integ', normals.shape)

    w, h = s
    normals = np.reshape(normals, (3,w,h))

    f_xs = np.empty((w,h))
    f_ys = np.empty((w,h))

    for u in range(w):
        
        for v in range(h):

            n1,n2,n3 = normals[:,u,v] 
            assert(np.isclose(np.linalg.norm(normals[:,u,v]), 1))
            
            f_x = n1 / n3
            f_y = n2 / n3

            f_xs[u,v] = f_x
            f_ys[u,v] = f_y



    f_xs = np.reshape(np.array(f_xs), s)
    f_ys = np.reshape(np.array(f_ys), s)
    z = utils.integrateFrankot(f_xs, f_ys)
    surface = np.reshape(z, s)
    # print('SURFACE', surface)
    
    
    return surface

    


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray (w,h)
        The depth map to be plotted

    Returns
    -------
        None

    """

    fig = plt.figure()
    # print('surface shape', surface.shape)
    ax = fig.add_subplot(111, projection='3d')
    xx, yy = np.mgrid[0:431, 0:369]
    # print(xx.shape, yy.shape)
    ax.plot_surface(xx, yy, surface, cmap=cm.coolwarm)
    plt.show()


if __name__ == '__main__':

    ##################################################################
    #                       1b 
    ##################################################################
    pxSize = 7e-6
    res = np.array([3840,2160])
    rad = 0.0075
    center = np.array([0,0,10])
    l1 = np.array([1,1,1]) / np.sqrt(3)
    l2 = np.array([1,-1,1]) / np.sqrt(3)
    l3 = np.array([-1,-1,1]) / np.sqrt(3)

    print('*************** q1a *********************')

    plot =  False

    renderNDotLSphere(center, rad, l1, pxSize, res, plot)
    renderNDotLSphere(center, rad, l2, pxSize, res, plot)
    renderNDotLSphere(center, rad, l3, pxSize, res, plot)


    print('*************** q1c *********************')
    I, L, s = loadData()

    print('*********  SVD  ***********')
    print('Performing SVD', I.shape)

    U, S, VH = np.linalg.svd(I,full_matrices=False)
    print(U.shape, S.shape, VH.shape)
    # S[3:] = 0
    # S_new  = np.eye(7,7)
    # np.fill_diagonal(S_new, S)

    # I = U @ S_new @ VH
    # print('After', I.shape)

    print('Singular values', S) 


    print('********** q1d: Estimate Pseudonormals Calibrated ****************')
    B = estimatePseudonormalsCalibrated(I, L)
        

    print('********** q1e: Estimate Albedos Normals **********')
    albedos, normals_scaled, normals = estimateAlbedosNormals(B)
    w,h = 431,369
    # (P), (3 x P)

    


    print('********** q1f: Estimate Albedos Normals **********')
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals_scaled, s)

    print('********** q1g: Normal integration **********')
    surface = estimateShape(normals, s)
   
    plotSurface(surface)

