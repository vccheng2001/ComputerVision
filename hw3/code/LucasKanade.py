import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

# top left, bot right 
# x0, y0
# x1, y1
def crop_img(img, rect):
    x1,y1,x2,y2 = rect

    return img[x1:x2, y1:y2]

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    
    p = p0

    # (x1,y1): top-left. (x2,h2): bottom-right
    x1, y1, x2,  y2 = rect
    template = crop_img(It, rect) # crop to rect
    img = It1
    # cv2.imshow('template', template)
    # cv2.waitKey()
    # cv2.imshow('img', img)
    # cv2.waitKey()
	
    
    print('rect', rect)

    w, h = img.shape[1], img.shape[0]
    for iter in range(int(num_iters)):
        print('p', p)
        # warp matrix 
        W = np.array([[1,0,p[0]],
                      [0,1,p[1]]])
        print('b4 warp', img.shape)
        warped_img = cv2.warpAffine(src=img, M=W, dsize=(w,h))
        print('warped', warped_img.shape)
        warped_img = crop_img(warped_img, rect)

        # (86, 35)
        print('cropped', warped_img.shape)

        # template - IWxp
        error = template - warped_img


        # gradient of image 
        gx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        gy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

        print('gx', gx.shape) # (240,320) h,w
        
        # compute gradient of image I, then warped onto
        # coordinate frame of T under current estimate of warp
        # W(x;p) (remember p is getting updated at each iteration)
        warped_gx = cv2.warpAffine(gx, W, dsize=(w,h))
        warped_gy = cv2.warpAffine(gy, W, dsize=(w,h))


        # # compute gradient of image
        # gx, gy = np.gradient(img)
        # print "gx =", gx
        # print "gy =", gy

        warped_gx = crop_img(warped_gx, rect)
        warped_gy = crop_img(warped_gy, rect)
        print('warped_gx', warped_gx.shape)
        print('warped_gy', warped_gy.shape)

        warped_grad = np.stack((warped_gx, warped_gy),axis=2)
        print('warped_grad', warped_grad.shape) # (86,35)
        # Jacobian dW/dp

        jacobian = np.array([[0,0,1], #(86,35,2)
                             [0,0,1]])

        # steepest descent images
        # (86,35,2) @ (2 x 3) = (86,35,3)
        sd = warped_grad @ jacobian
        
        print('steepest descent', sd.shape)
        sd_T = sd.T
        # (3,35,86)
        print('steepest descent T', sd_T.shape)

        print('sd_T @ sd', (sd_T @ sd).shape)
        # Hessian
        H = (sd_T @ sd).sum((0,1))
        
        # steepest descent parameter updates computed by
        # dot proding error with steepest descent images

        sd_updates = (sd_T @ error).sum((0,1))
        # get final parameter updates
        dp = np.linalg.inv(H) @ sd_updates
        p = p + dp

        if np.linalg.norm(dp)**2 < threshold:
            break
    
    return p
