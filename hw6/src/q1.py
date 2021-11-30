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
from utils import integrateFrankot
import scipy
import scipy.io
from PIL import Image
import matplotlib.cm as cm
import cv2

def renderNDotLSphere(center, rad, light, pxSize, res):

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
    plt.imshow(ndotl.astype('float64'), cmap="gray")
    plt.show()

    return ndotl



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

renderNDotLSphere(center, rad, l1, pxSize, res)
renderNDotLSphere(center, rad, l2, pxSize, res)
renderNDotLSphere(center, rad, l3, pxSize, res)


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

    im1 = np.array(Image.open(f'{path}input_1.tif'), dtype="uint16") 
    s = (im1.shape[0], im1.shape[1])
    I = im1.flatten()

    # stack remaining imgs
    for i in range(1,7):
        im = np.array(Image.open(f'{path}input_{i+1}.tif'), dtype="uint16")
        im = im.flatten()
        I = np.vstack((I,im))

    L = np.load(f'{path}/sources.npy').T # (3x7)
    print('L', L.shape) # (3,7)
    print('I', I.shape) # (7, 477117)
    print('s', s)       # (431, 369)

    return I, L, s



##################################################################
#                       1c
##################################################################
print('*************** q1c *********************')
loadData()


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

    # B = # (3xP), set of pseudonormals in image
    I = L.T @ B # rank of I = 3
    U, S, VH = np.linalg.svd(I, full_matrices=False)
    I_svd = VH[8,:]

    # Divide by scale
    I_svd = (I_svd / I_svd[-1]).reshape(3,3)

    B = None
    return B


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

    n_tilde = B 
    magnitude = np.linalg.norm(n_tilde) # magnitude
    albedos = magnitude # albedo

    # NORMALS, surface normal is derivative of depth
    normals = n_tilde / magnitude

    return albedos, normals


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

    albedoIm = np.reshape(albedos, s)
    normalIm = np.reshape(normals, (s,3))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    nx = normalIm[:,0]
    ny = normalIm[:,1]
    nz = normalIm[:,2]

    surf = ax.scatter(nx,ny,nz, cmap=cm.rainbow,
                           linewidth=0)
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

    # [N ] [V ] = 0 (nx3) by ()

    # N dot V1 = 0
    # N dot V2 = 0
    # ....
    # N dot VN = 0
    _, P = normals.shape

    for i in range(P): # loop through points
        N = normals[:,i] # (3x1)
        nx, ny, nz = N
        V[i] = [1, 0, z[x+1,y] - zxy]



    # estimate z values 

    surface = None
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    pass


if __name__ == '__main__':

    # Put your main code here
    pass
