import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2



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
    x1, y1, x2, y2 = rect
    # width, height of rect 
    rect_w, rect_h = int(x2-x1), int(y2-y1)

    # gradient of img
    img = It1
    template = It
    img_h, img_w = img.shape

	
    
    y = np.arange(0, img_h, 1)
    x = np.arange(0, img_w, 1)
    rect_x = np.linspace(x1, x2, rect_w)
    rect_y = np.linspace(y1, y2, rect_h)
    img_grid_x, img_grid_y  = np.meshgrid(rect_x, rect_y)
    
    
    I_gx, I_gy = np.gradient(img)
    
    # splines 
    sp_template = RectBivariateSpline(y, x, template)
    sp_img = RectBivariateSpline(y, x, img)
    sp_gx = RectBivariateSpline(y, x, I_gx)
    sp_gy = RectBivariateSpline(y, x, I_gy)

    # crop to rectangle
    template = sp_template.ev(img_grid_y,img_grid_x)


    for iter in range(int(num_iters)):

        #  (1) warped img by translation 
        x1_w, y1_w = x1+p[0], y1+p[1]
        x2_w, y2_w = x2+p[0], y1+p[1]   


        # Evaluate in rect
        # warped x, y coords in rectangle
        wp_rect_x = np.linspace(x1_w, x2_w, rect_w)
        wp_rect_y = np.linspace(y1_w, y2_w, rect_h)
        wp_img_grid_x, wp_img_grid_y = np.meshgrid(wp_rect_x, wp_rect_y)
        # print('wp_img_grid_x', wp_img_grid_x.shape)
        # evaluate at warped locations 
        warped_img = sp_img.ev(wp_img_grid_y, wp_img_grid_x)   

        # print('wp_img', warped_img.shape)   
        # print('template', template.shape)
        # (2) compute error Template - Warped Image
        error = template - warped_img
        # print('error', error.shape) # (86,35)

        # (3) warp gradient with W(x,p)
        # first, calculate gradient

        # (4) evaluate jacobian at warped image locs
        wp_grad_x = sp_gx.ev(wp_img_grid_y, wp_img_grid_x).flatten() 
        wp_grad_y = sp_gy.ev(wp_img_grid_y, wp_img_grid_x).flatten()

        # print('warped_grad_x', wp_grad_x.shape) # (86,35)
        warped_grad = np.stack((wp_grad_x, wp_grad_y)).T

        # (5) Steepest descent deltaI @ dW/dp
        # print('warped_grad', warped_grad.shape)
        sd = warped_grad @ np.array([[1,0],[0,1]])
        sd_T = sd.T
        # print('sd_T', sd_T.shape)

        # (6) Compute Hessian
        H = sd_T @ sd
      
        # (7, 8) Compute dp
        # (2x2) @ (2xN)@(Nx1) = (2x1)
        dp = np.linalg.inv(H) @ (sd_T @ error.flatten())

        # print('dp', dp.shape)
        # print('p', p.shape)
        # (9) Update parameters
       
        p = p + dp
        # Check stopping condition 
        if np.linalg.norm(dp)**2 < threshold:
            break


    
    return p






def LucasKanadeWTC(It, It1, rect, threshold, num_iters, p0=np.zeros(2), update_template=True):
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
    x1, y1, x2, y2 =map(int, rect)
    # width, height of rect 
    rect_w, rect_h = x2-x1,y2-y1

    # gradient of img
    img = It1
    template = It
    img_h, img_w = img.shape

	
    
    y = np.arange(0, img_h, 1)
    x = np.arange(0, img_w, 1)
    rect_x = np.linspace(x1, x2, rect_w)
    rect_y = np.linspace(y1, y2, rect_h)
    img_grid_x, img_grid_y  = np.meshgrid(rect_x, rect_y)
    
    I_gx, I_gy = np.gradient(img)
    
    # splines 
    sp_img = RectBivariateSpline(y, x, img)
    sp_gx = RectBivariateSpline(y, x, I_gx)
    sp_gy = RectBivariateSpline(y, x, I_gy)

    # original template: region of interest to track

    if update_template:
        sp_template = RectBivariateSpline(y, x, template)
        template = sp_template.ev(img_grid_y,img_grid_x)
    

    for iter in range(int(num_iters)):

        #  (1) warped img by translation 
        x1_w, y1_w = x1+p[0], y1+p[1]
        x2_w, y2_w = x2+p[0], y1+p[1]   


        # Evaluate in rect
        # warped x, y coords in rectangle
        wp_rect_x = np.linspace(x1_w, x2_w, rect_w)
        wp_rect_y = np.linspace(y1_w, y2_w, rect_h)
        
        wp_img_grid_x, wp_img_grid_y = np.meshgrid(wp_rect_x, wp_rect_y)
        # evaluate at warped locations 
        warped_img = sp_img.ev(wp_img_grid_y, wp_img_grid_x)   

        # print('wp_img', warped_img.shape)   
        # print('template', template.shape)
        # (2) compute error Template - Warped Image
        error = template - warped_img
        # print('error', error.shape) # (86,35)

        # (3) warp gradient with W(x,p)
        # first, calculate gradient

        # (4) evaluate jacobian at warped image locs
        wp_grad_x = sp_gx.ev(wp_img_grid_y, wp_img_grid_x).flatten() 
        wp_grad_y = sp_gy.ev(wp_img_grid_y, wp_img_grid_x).flatten()

        # print('warped_grad_x', wp_grad_x.shape) # (86,35)
        warped_grad = np.stack((wp_grad_x, wp_grad_y)).T

        # (5) Steepest descent deltaI @ dW/dp
        # print('warped_grad', warped_grad.shape)
        sd = warped_grad @ np.array([[1,0],[0,1]])
        sd_T = sd.T
        # print('sd_T', sd_T.shape)

        # (6) Compute Hessian
        H = sd_T @ sd
      
        # (7, 8) Compute dp
        # (2x2) @ (2xN)@(Nx1) = (2x1)
        dp = np.linalg.inv(H) @ (sd_T @ error.flatten())

        # print('dp', dp.shape)
        # print('p', p.shape)
        # (9) Update parameters
        p_prev = p
        p = p + dp

    

        # Check stopping condition 
        if np.linalg.norm(dp)**2 < threshold:
            break

    epsilon = threshold
    if np.linalg.norm(p-p_prev) > epsilon:
        # print("Don't update")
        prev_template = template # don't update
    else: 
        prev_template = None

            
    return p, prev_template
