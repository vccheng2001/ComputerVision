from cv2 import warpAffine
import numpy as np
from LucasKanadeAffine import *
from InverseCompositionAffine import *
def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t (template)
    :param image2: Images at time t+1 (image)
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """


    # src: coordinates in the source image
    # dst: coordinates in the output image
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters, p0 = np.zeros(6))

    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    transf = warpAffine(image1, M, dsize=(image2.shape[1],image2.shape[0]))
    

    diff =  np.abs(image2 - transf)

    # mask extracts "moving" objects
    _, mask = cv2.threshold(diff, tolerance, 255, cv2.THRESH_BINARY_INV)
    image1 = (image1* 255).astype(np.uint8)

    # convert to rgb, mark moving points as red 
    result = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    idx = np.where(mask==0)
    result[idx[0],idx[1],:] = [0, 0, 255]


    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return result


#     return result
# # a mask only considers pixels in the original image where the mask is greater than zero.
def apply_mask(frame, mask):
    """Apply binary mask to frame, return in-place masked image."""
    return cv2.bitwise_or(frame, frame, mask=mask)
