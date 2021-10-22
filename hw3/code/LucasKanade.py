import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

# # cropping a rectangle from image
# # returns grid x, grid y
# def create_grid_at_rect(rect, rect_w, rect_h):
#     x1, y1, x2, y2 = map(int, rect)
#     rect_x = np.linspace(x1, x2, rect_w)
#     rect_y = np.linspace(y1, y2, rect_h)
#     img_grid_x, img_grid_y  = np.meshgrid(rect_x, rect_y,indexing='ij')
#     return img_grid_x, img_grid_y



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
    dp = np.array([1,1])# set to initial value

    # (x1,y1): top-left. (x2,h2): bottom-right
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]


    # setup
    template = It
    img = It1
    img_h, img_w = img.shape
    x = np.arange(0, img_h, 1)
    y = np.arange(0, img_w, 1)


    # create rect grid 
    rect_w, rect_h = int(x2-x1), int(y2-y1)
    # crop to rect
    rect_x = np.linspace(x1, x2, rect_w) # go down rows
    rect_y = np.linspace(y1, y2, rect_h)
    rect_grid_x, rect_grid_y  = np.meshgrid(rect_x, rect_y)
    
    
    # gradient 
    I_gy, I_gx = np.gradient(img)

    
    # splines 
    sp_img = RectBivariateSpline(x,y, img)
    sp_template = RectBivariateSpline(x,y, template)
    sp_gx = RectBivariateSpline(x,y, I_gx)
    sp_gy = RectBivariateSpline(x,y, I_gy)

    # crop template to rect
    T = sp_template.ev(rect_grid_y,rect_grid_x)


    for iter in range(int(num_iters)):

        if np.linalg.norm(dp) ** 2 < threshold:
            break

        #  (1) warped img by translation 
        x1_w, y1_w, x2_w, y2_w = x1+p[0], y1+p[1], x2+p[0], y2+p[1]   



        # warped rect
        wp_rect_x = np.linspace(x1_w, x2_w, rect_w)
        wp_rect_y = np.linspace(y1_w, y2_w, rect_h)
        wp_rect_grid_x, wp_rect_grid_y = np.meshgrid(wp_rect_x, wp_rect_y)

        # crop image to warped rect
        warped_img = sp_img.ev(wp_rect_grid_y, wp_rect_grid_x)   

 
        # (2) compute error Template - Warped Image
        error = T - warped_img
        error = error.reshape(-1,1)

        # (3) warp gradient with W(x,p)
        # first, calculate gradient

        # (4) evaluate jacobian at warped image locs
        wp_grad_x = sp_gx.ev(wp_rect_grid_y, wp_rect_grid_x).flatten() 
        wp_grad_y = sp_gy.ev(wp_rect_grid_y, wp_rect_grid_x).flatten()

        warped_grad = np.stack((wp_grad_x, wp_grad_y)).T

        # (5) Steepest descent deltaI @ dW/dp
        sd = warped_grad @ np.array([[1,0],[0,1]])
        sd_T = sd.T

        # (6) Compute Hessian
        H = sd_T @ sd
      
        # (7, 8) Compute dp
        dp = np.linalg.inv(H) @ sd_T @ error

        # (9) Update parameters
        p[0] = p[0] + dp[0,0]
        p[1] = p[1] + dp[1,0]


    
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
    dp = np.array([1,1])# set to initial value
    x1, y1, x2, y2 = map(int, rect)
    
    # setup
    template = It
    img = It1
    img_h, img_w = img.shape
    x = np.arange(0, img_h, 1)
    y = np.arange(0, img_w, 1)


    # create rect grid 
    rect_w, rect_h = int(x2-x1), int(y2-y1)
    # crop to rect
    rect_x = np.linspace(x1, x2, rect_w)
    rect_y = np.linspace(y1, y2, rect_h)
    rect_grid_x, rect_grid_y  = np.meshgrid(rect_x, rect_y)
    
    # gradient 
    I_gy, I_gx = np.gradient(img)

    
    # splines 
    sp_img = RectBivariateSpline(x,y, img)
    sp_gx = RectBivariateSpline(x,y, I_gx)
    sp_gy = RectBivariateSpline(x,y, I_gy)

    # original template: region of interest to track
    if update_template:
        sp_template = RectBivariateSpline(x,y,template)
        # crop template to rect
        T = sp_template.ev(rect_grid_y,rect_grid_x)    
    else: 
        T = template

    for iter in range(int(num_iters)):

        if np.linalg.norm(dp) ** 2 < threshold:
            break


        #  (1) warped img by translation 
        x1_w, y1_w, x2_w, y2_w = x1+p[0], y1+p[1], x2+p[0], y2+p[1]   



        # warped rect
        wp_rect_x = np.linspace(x1_w, x2_w, rect_w)
        wp_rect_y = np.linspace(y1_w, y2_w, rect_h)
        wp_rect_grid_x, wp_rect_grid_y = np.meshgrid(wp_rect_x, wp_rect_y)

        # crop image to warped rect
        warped_img = sp_img.ev(wp_rect_grid_y, wp_rect_grid_x)   

 
        # (2) compute error Template - Warped Image
        error = T- warped_img

        error = error.reshape(-1,1)

        # (3) warp gradient with W(x,p)
        # first, calculate gradient

        # (4) evaluate jacobian at warped image locs
        wp_grad_x = sp_gx.ev(wp_rect_grid_y, wp_rect_grid_x).flatten() 
        wp_grad_y = sp_gy.ev(wp_rect_grid_y, wp_rect_grid_x).flatten()

        warped_grad = np.stack((wp_grad_x, wp_grad_y)).T

        # (5) Steepest descent deltaI @ dW/dp
        sd = warped_grad @ np.array([[1,0],[0,1]])
        sd_T = sd.T

        # (6) Compute Hessian
        H = sd_T @ sd
      
        # (7, 8) Compute dp
        dp = np.linalg.inv(H) @ sd_T @ error

        p_prev = p
        # (9) Update parameters
        p[0] = p[0] + dp[0,0]
        p[1] = p[1] + dp[1,0]


    epsilon = threshold
    if np.linalg.norm(p-p_prev) > epsilon:
        prev_template = template # don't update
    else: 
        prev_template = None

            
    return p, prev_template
