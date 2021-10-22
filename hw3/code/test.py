import scipy 
import cv2
from scipy.interpolate import RectBivariateSpline
import numpy as np

img = np.random.rand(15,10)
img_h, img_w = img.shape
y = np.arange(0, img_h, 1)
x = np.arange(0, img_w, 1)

print('x', x)
print('y', y)

rect = 2,5,4,6
x1,x2,y1,y2 = rect
rect_w, rect_h = int(abs(x2-x1)), int(abs(y2-y1))

rect_x = np.linspace(x1, x2, rect_w)
print('rx', rect_x)
rect_y = np.linspace(y1, y2, rect_h)
print('ry', rect_y)

img_grid_x, img_grid_y  = np.meshgrid(rect_x, rect_y, indexing="ij")

print('gridx', img_grid_x)
print('gridy', img_grid_y)

sp_img = RectBivariateSpline(y, x, img)


# crop to rectangle
imgg = sp_img.ev(img_grid_y,img_grid_x)
print(imgg)