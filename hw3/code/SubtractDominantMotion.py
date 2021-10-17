from cv2 import warpAffine
import numpy as np
from LucasKanadeAffine import *
def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """


    # src: coordinates in the source image
    # dst: coordinates in the output image
    M = LucasKanadeAffine(image1, image2, threshold, num_iters, p0 = np.zeros(6))

    transf = warpAffine(image1, M, dsize=(image2.shape[1],image2.shape[0]))
    
    cv2.imshow('transf', transf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    diff =  np.abs(image2 - image1)#transf)

    tolerance=0.05
    image1 = image1.astype(np.uint8)
    print(diff)
    # if  > threshold, set to black, else white 
    mask = cv2.threshold(diff, tolerance, 255, cv2.THRESH_BINARY_INV)[1]
    mask = mask.astype(np.uint8)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Writing result...')


    result = apply_mask(image1, mask)
    cv2.imwrite("result.jpg", result)

    return result
# a mask only considers pixels in the original image where the mask is greater than zero.
def apply_mask(frame, mask):
    """Apply binary mask to frame, return in-place masked image."""
    return cv2.bitwise_and(frame, frame, mask=mask)
