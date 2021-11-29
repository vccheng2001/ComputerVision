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


def renderNDotLSphere(center, rad, lights, pxSize, res):

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
    cx, cy, cz = center
    # sphere: sqrt(x^2 + y^2 + z^2) = c
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)


    # conversion into cartesian coords:
    # x = r * cos(theta) * sin(phi)
    # y = r * sin(theta) * sin(phi)
    # z = r * cos(phi)

    x = rad * np.outer(np.cos(u), np.sin(v))
    y = rad * np.outer(np.sin(u), np.sin(v))
    z = rad * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x-cx, y-cy, z-cz, color='b')

    plt.show()

    # normals 
    nx = x.flatten() - cx
    ny = y.flatten() - cy 
    nz = z.flatten() - cz 

    normals = np.vstack((nx,ny,nz))
    normals /= np.linalg.norm(normals, axis=0)

    print('normals', normals[:,1]) # 3 x N

    ro = 1
    # I: Nx1
    # lights: Nx3
    # n: 3x1 
    S = lights # not invertible if linear combination 
    # between light sources. So lights can't lie in same plane
    I = np.max(0, S@(ro*normals))
    
    # kc/π = 1
    # I = (ρ/π)kccos(theta)=ρcos(theta)
    # I = ρs^T n
    n_tilde = np.linalg.inv(S.T@S)@S.T@I
    magnitude = np.linalg.norm(n_tilde) # magnitude
    ro = magnitude # albedo

    # NORMALS, surface normal is derivative of depth
    n = n_tilde / magnitude

    I = S @ (ro*normals)
    print('ro', ro)
    print('n', n[:,1])



    # Depth from normals
    # assume orthographic camera
    # goal: estimate z at xy

    # how to draw normal of surface?

    
    image = None
    return image


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

# lights = np.vstack([l1,l2,l3]) # 3x3
# renderNDotLSphere(center, rad, lights, pxSize, res)



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
    print()

    return I, L, s



##################################################################
#                       1c
##################################################################
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

    albedos = None
    normals = None
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

    albedoIm = None
    normalIm = None

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
