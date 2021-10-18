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
    

    diff =  np.abs(image2 - image1)#transf)

    tolerance=0.05
    image1 = image1.astype(np.uint8)


    _, mask = cv2.threshold(diff, tolerance, 255, cv2.THRESH_BINARY_INV)
    result = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    color= cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB).astype(np.uint8)

    result[mask==0] = (255,0,0)
    result[mask!=0] = (255,255,255)
    cv2.imshow("result", color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result = cv2.addWeighted(result, 0.5, color,0.5, 0)



    cv2.imshow("result w", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("result.jpg", result)


    # # if  > threshold, set to black, else white 
    # _, mask = cv2.threshold(diff, tolerance, 255, cv2.THRESH_BINARY_INV)
    # color =  cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    # result = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    
    # result[mask==0] = (255,0,0)

    # result = apply_mask(color, result).astype(np.uint8)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("result.jpg", result)
    # return 

#     return result
# # a mask only considers pixels in the original image where the mask is greater than zero.
def apply_mask(frame, mask):
    print('f', frame.shape, mask.shape, type(frame), type(mask))
    """Apply binary mask to frame, return in-place masked image."""
    return cv2.bitwise_or(frame, frame, mask=mask)
