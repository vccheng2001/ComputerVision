import numpy as np
from scipy.interpolate import RectBivariateSpline

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
    template = It
    img = It1

    for iter in range(num_iters):
        x1_wp, y1_wp, x2_wp, y2_wp = x1+p0[0], y1+p0[1], x2+p0[0], y2+p0[1]


        # warp I with W(x;p) = x+p to get I(W(x;p))
        warped_img = img + p # x'
        
        # template - IWxp
        error = template - warped_img

        # warp gradient dI with W(x;p)
        # warped gradients: delta_I, eval at W(x,p)
        warped_grad = np.gradient(img) + p

        # Jacobian dW/dp

        jacobian = np.array([1,1])

        # steepest descent images
        sd = warped_grad @ jacobian

        # Hessian
        H = sd.T @ sd
        
        # dp

        dp = np.linalg.inv(H) @ sd.T @ error
        p = p + dp

        if np.linalg.norm(p)**2 < threshold:
            break
    
    return p
