import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """


    p = np.array([[1,0,0],[1,0,0]]).astype(np.float32)


    img, template = It1, It
    h, w = template.shape
    y = np.arange(0, h, 1)
    x = np.arange(0, w, 1)
    img_grid_x, img_grid_y  = np.meshgrid(x,y)

    # gradient
    T_gx, T_gy = np.gradient(template)

    # splines 
    sp_template = RectBivariateSpline(y, x, template)
    sp_Tgx = RectBivariateSpline(y, x, T_gx)
    sp_Tgy = RectBivariateSpline(y, x, T_gy)

    # evaluate template at image 
    template = sp_template.ev(img_grid_y,img_grid_x)


    # (4) evaluate gradient of template deltT
    # template grid 
    wp_grad_x = sp_Tgx.ev(img_grid_y, img_grid_x).flatten() 
    wp_grad_y = sp_Tgy.ev(img_grid_y, img_grid_x).flatten()
    warped_grad = np.stack((wp_grad_x, wp_grad_y)).T

    # calculate steepest parameter update 
    # delt T @ dW/dp
    sd = calc_sd_affine(warped_grad, template)
    sd_T = sd.T

    # (6) Compute Hessian
    H = sd_T @ sd
      
  
    for iter in range(int(num_iters)):

        # computes an inverse affine transformation represented by 2×3 matrix M:
        # M0,M1,M2,M3,M4,M5= M.flatten()
        p_inv = cv2.invertAffineTransform(p)

        warped_img = cv2.warpAffine(img.copy(), p_inv.copy(), dsize=(template.shape[1], template.shape[0]))

        # # (2) compute error 
        # warped image depends on p 
        # p gets updated each iter
        error = warped_img - template

      
        # (7, 8) Compute dp
        # hessian independent of warp parameters p
        dp = np.linalg.inv(H) @ (sd_T @ error.flatten())
        dp = dp.reshape(p.shape)

        
        # (9) Update parameters
        p_matrix = np.vstack((p, [0, 0, 1]))
        dp_matrix = np.eye(3) + np.vstack((dp, [0,0,0]))
        p = p_matrix @ (np.linalg.inv(dp_matrix))
        # get first two rows only
        p = p[:2,:]

        # Check stopping condition 
        if np.linalg.norm(dp)**2 < threshold:
            print("Done")
            break

    return p

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

