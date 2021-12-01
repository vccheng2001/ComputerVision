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

    # first image 
    im = cv2.imread(f'{path}input_1.tif', cv2.IMREAD_UNCHANGED)
    assert(im.dtype == np.uint16)
    # extract luminance, flatten into vector
    lum = skimage.color.rgb2gray(im)
    s = lum.shape
    lum = lum.flatten()

    I = lum
    
    # W(431)*H(369) = 159039 PIXELS * 3 CHANNELS = 477117
    # stack remaining images 
    for i in range(1,7):
        im = cv2.imread(f'{path}input_{i+1}.tif', cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        assert(im.dtype == np.uint16)
        lum = skimage.color.rgb2gray(im)
        lum = lum.flatten()
        I = np.vstack((I,lum))

    # divide by max to get relative luminance (range 0->1)
    I = I / np.max(I)

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
    
    # METHOD 1: CLOSED FORM 

    L_inv = L.T @ np.linalg.inv(L @ L.T) # 490, 574, 547
    b1 = L_inv.T @ I 


    # METHOD 2: LEAST SQUARES

    _, P = I.shape 
    identity = scipy.sparse.eye(P)

    # multiply by identity matrix PxP
    L = sparse.kron(L,identity).T
    I = np.reshape(I, (7*P))
    I = sparse.csr_matrix(I).T
    
    # L: M=7p, N=3p
    # I: M=7p
    # B: N=3p

    # solve least squares using sparse matrix 
    ret = sparse.linalg.lsqr(L, I.toarray())
    b2 = ret[0]
    b2 = np.reshape(b2, (3, -1))

    return b2




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

    for u in range(w):
        for v in range(h):
            # for each pixel, normalize to unit length
            norm, unit_vec = normalize(n_tilde[:, u,v])
            # albedos: magnitude 
            albedos.append(norm)

            # ranges [-1,1]
            normals[:,u,v] = unit_vec

    # convert from [-1,1] to [0,1] range
    normals_scaled = ((normals + 1)/2)
    normals_scaled = np.reshape(normals_scaled, (3, P))
    albedos = np.array(albedos)

    # scale to between 0,1
    albedos = scale_zero_to_one(albedos)
    
    return albedos, normals_scaled, normals
  


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



    # reshape albedos into image shape
    albedoIm = np.reshape(albedos, (w,h))
    normalIm = np.reshape(normals, (w,h,3))

    # plot albedo
    fig = plt.figure()
    plt.imshow(albedoIm, cmap=cm.gray)
    plt.show()

    # plot normals
    fig = plt.figure()
    plt.imshow(normalIm, cmap=cm.rainbow)
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

    # loop through each pixel
    for u in range(w):

        for v in range(h):
            # calculate normals (should be unit vector)
            n1,n2,n3 = normals[:,u,v] 
            assert(np.isclose(np.linalg.norm(normals[:,u,v]), 1))
            
            f_x = n1 / n3
            f_y = n2 / n3

            # gradients
            f_xs[u,v] = f_x
            f_ys[u,v] = f_y

    f_xs = np.reshape(np.array(f_xs), s)
    f_ys = np.reshape(np.array(f_ys), s)
    # integrate to estimate shape
    z = utils.integrateFrankot(f_xs, f_ys)
    surface = np.reshape(z, s)
    
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

    print('Singular values', S) 


    print('********** q1d: Estimate Pseudonormals Calibrated ***********')
    B = estimatePseudonormalsCalibrated(I, L)
        

    print('********** q1e: Estimate Albedos Normals **********')
    albedos, normals_scaled, normals = estimateAlbedosNormals(B)

    print('********** q1f: Estimate Albedos Normals **********')
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals_scaled, s)

    print('********** q1g: Normal integration **********')
    surface = estimateShape(normals, s)
   
    plotSurface(surface)

