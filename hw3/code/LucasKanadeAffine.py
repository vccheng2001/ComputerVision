import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import cv2


def calc_sd_affine(warped_grad, template):
    # (5) Steepest descent deltaI @ dW/dp
    x2,y2 = template.shape[1], template.shape[0]
    # 256, 256
    sd_size = (y2,x2, 6)
    # (1x2) @ (2x6) for each pixel
    sd = np.zeros(sd_size)
    warped_grad = np.reshape(warped_grad,(y2,x2,2))
    for r in range(y2): # loop through template height (rows)
        for c in range(x2): # loop cols

            # for each pixel located at r,c
            deltI = np.expand_dims(warped_grad[r][c], 0)
            

            # 1 x 2 @ 2 x  6 = 1 x 6
            sd[r,c] = np.expand_dims(deltI @ np.array([[c,0,r,0,1,0],
                                          [0,c,0,r,0,1]]), 0)

            
    sd = sd.reshape((x2*y2,6))
    
    return sd



def LucasKanadeAffine(It, It1, threshold, num_iters, p0 = np.zeros(6)):
    """
    #     :param It: template image
    #     :param It1: Current image
    #     :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    #     :param num_iters: number of iterations of the optimization
    #     :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    #    
    """

    
    p = p0
    # Warp matrix
    M = np.array([[1+p[0], p[1], p[2]], 
                  [p[3], 1+p[4], p[5]]])

    img, template = It1, It
    h, w = template.shape


    y = np.arange(0, h, 1)
    x = np.arange(0, w, 1)
    img_grid_x, img_grid_y  = np.meshgrid(x,y)

    # gradient
    I_gx, I_gy = np.gradient(img)

    # splines 
    sp_template = RectBivariateSpline(y, x, template)
    sp_gx = RectBivariateSpline(y, x, I_gx)
    sp_gy = RectBivariateSpline(y, x, I_gy)

    # evaluate template at image 
    template = sp_template.ev(img_grid_y,img_grid_x)
  
    for iter in range(int(num_iters)):

        M =  np.array([[1+p[0], p[1], p[2]], 
                  [p[3], 1+p[4], p[5]]])


        M0,M1,M2,M3,M4,M5 = M.flatten()

        warped_img = cv2.warpAffine(img.copy(), M.copy(), dsize=(template.shape[1],template.shape[0]))

        # (2) compute error Template - Warped Image
        error = template - warped_img

        # (3) warp gradient with W(x,p)
        x1,y1= 0,0
        x2,y2 = template.shape[1], template.shape[0]
        x1_w, y1_w = M0*x1+M1*y1+M2, M3*x1+M4*y1+M5
        x2_w, y2_w = M0*x2+M1*y2+M2, M3*x2+M4*y2+M5
        wp_rect_x = np.linspace(x1_w, x2_w, y2)
        wp_rect_y = np.linspace(y1_w, y2_w, x2)
        wp_img_grid_x, wp_img_grid_y = np.meshgrid(wp_rect_x, wp_rect_y)
        

        # (4) evaluate jacobian at warped image locs
        wp_grad_x = sp_gx.ev(wp_img_grid_y, wp_img_grid_x).flatten() 
        wp_grad_y = sp_gy.ev(wp_img_grid_y, wp_img_grid_x).flatten()
        warped_grad = np.stack((wp_grad_x, wp_grad_y)).T

        # calculate steepest parameter update 
        sd = calc_sd_affine(warped_grad, template)
        sd_T = sd.T

        # (6) Compute Hessian
        H = sd_T @ sd
      
        # (7, 8) Compute dp
        dp = np.linalg.inv(H) @ (sd_T @ error.flatten())

        # (9) Update parameters
        p = p + dp
        
        # Check stopping condition 
        if np.linalg.norm(dp)**2 < threshold:
            break


        M = np.array([[1+p[0], p[1], p[2]], 
                  [p[3], 1+p[4], p[5]]])

    
    return M
