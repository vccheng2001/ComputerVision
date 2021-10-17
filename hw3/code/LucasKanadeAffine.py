import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import cv2



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

    # initial guess: p = 0
    # M is equivalent to W(x,p)
    M = np.array([[1+p[0], p[1], p[2]], 
                  [p[3], 1+p[4], p[5]]])
    img, template = It1, It

    # x1 = 0, x2 = w
    # y1 = 0, y2 = h
    h, w = template.shape



    y = np.arange(0, h, 1)
    x = np.arange(0, w, 1)
    rect_x = np.linspace(0, w, w)
    rect_y = np.linspace(0, h, h)
    img_grid_x, img_grid_y  = np.meshgrid(rect_x, rect_y)


    I_gx, I_gy = np.gradient(img)

    # splines 
    sp_template = RectBivariateSpline(y, x, template)
    sp_img = RectBivariateSpline(y, x, img)
    sp_gx = RectBivariateSpline(y, x, I_gx)
    sp_gy = RectBivariateSpline(y, x, I_gy)

    template = sp_template.ev(img_grid_y,img_grid_x)





    print('p0', p0)    
    for iter in range(int(num_iters)):

        M =  np.array([[1+p[0], p[1], p[2]], 
                  [p[3], 1+p[4], p[5]]])


        M0,M1,M2,M3,M4,M5= M.flatten()

        # x' = M00x + M01y  + b00
        # y' = M10x + M11y + b10

        # scipy.ndimage.affine_transform(img, M)
        warped_img = cv2.warpAffine(img.copy(), M.copy(), dsize=(template.shape[1],template.shape[0]))
        # 320, 240

        # (2) compute error Template - Warped Image
        error = template - warped_img

        # (3) warp gradient with W(x,p)
        x1,y1=0,0
        x2,y2 = template.shape[1], template.shape[0]

        x1_w, y1_w = M0*x1+M1*y1+M2, M3*x1+M4*y1+M5
        x2_w, y2_w = M0*x2+M1*y2+M2, M3*x2+M4*y2+M5


        # Evaluate in rect
        # warped x, y coords in rectangle
        wp_rect_x = np.linspace(x1_w, x2_w, template.shape[1])
        wp_rect_y = np.linspace(y1_w, y2_w, template.shape[0])
        wp_img_grid_x, wp_img_grid_y = np.meshgrid(wp_rect_x, wp_rect_y)
        # print('wp_img_grid_x', wp_img_grid_x.shape)
        # evaluate at warped locations 
        # warped_img = sp_img.ev(wp_img_grid_y, wp_img_grid_x)   

        


        # (4) evaluate jacobian at warped image locs
        wp_grad_x = sp_gx.ev(wp_img_grid_y, wp_img_grid_x).flatten() 
        wp_grad_y = sp_gy.ev(wp_img_grid_y, wp_img_grid_x).flatten()

        # print('warped_grad_x', wp_grad_x.shape) # (86,35)
        # (Nx2)
        warped_grad = np.stack((wp_grad_x, wp_grad_y)).T

        # (5) Steepest descent deltaI @ dW/dp
        # (1x2) @ (2x6) for each pixel
        sd= np.zeros((It.shape[0]*It.shape[1], 6))
        for j in range(y2): # rows
            for k in range(x2): #cols
                I = warped_grad[j*x2+k]
                J = np.array([[k,0,j,0,1,0],
                              [0,k,0,j,0,1]])

                sd[j*x2+k] = I @ J
        sd_T = sd.T

        # (6) Compute Hessian
        H = sd_T @ sd
      
        # (7, 8) Compute dp
        # (2x2) @ (2xN)@(Nx1) = (2x1)
        dp = np.linalg.inv(H) @ (sd_T @ error.flatten())


        # (9) Update parameters
       
        p = p + dp
        # Check stopping condition 
        if np.linalg.norm(dp)**2 < threshold:
            print('below threshold')
            break


        M = np.array([[1+p[0], p[1], p[2]], 
                  [p[3], 1+p[4], p[5]]])

    
    return M





# # def LucasKanadeAffine(It, It1, threshold, num_iters):
# #     """
# #     :param It: template image
# #     :param It1: Current image
# #     :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
# #     :param num_iters: number of iterations of the optimization
# #     :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
# #     """

# #     # put your implementation here
# #     M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
# #     return M
