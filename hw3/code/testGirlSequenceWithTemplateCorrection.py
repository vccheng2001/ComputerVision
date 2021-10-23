import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import *





parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=0.01, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

temp = seq[:,:,0] * 255
x1, y1, x2, y2 = map(int, rect)
# setup
temp_h, temp_w = temp.shape
x = np.arange(0, temp_h, 1)
y = np.arange(0, temp_w, 1)
# create rect grid 
rect_w, rect_h = int(x2-x1), int(y2-y1)
# crop to rect
rect_x = np.linspace(x1, x2, rect_w) # go down rows
rect_y = np.linspace(y1, y2, rect_h)
rect_grid_x, rect_grid_y  = np.meshgrid(rect_x, rect_y)


sp_template = RectBivariateSpline(x,y, temp)
init_template = sp_template.ev(rect_grid_y,rect_grid_x)


h, w, num_frames = seq.shape
rects = []
rects_wtc = []


for i in range(1, num_frames-1):
    print(f'Processing frame {i}')
    
    
    in_template = seq[:,:,i-1] * 255
    img = seq[:,:,i] * 255
   

    # find offset from i-1 to i
    p = LucasKanade(in_template, img, rect, threshold, num_iters, p0=np.zeros(2))

    rect[0] += p[0] # adding n-1 -> n
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    rects.append(rect)

    # find offset from frame 1 to i 
    # how much to add to initial bounding box to get from initial to current image 
    ptot = LucasKanade(init_template, img, rect, threshold, num_iters, p0=np.zeros(2),use_template=False)
    rect[0] += ptot[0] 
    rect[1] += ptot[1]
    rect[2] += ptot[0]
    rect[3] += ptot[1]

    rect_wtc = rect.copy()
    rects_wtc.append(rect_wtc)
    

        
    if i == 1 or i % 20 == 0:
        plt.figure()
        fig, ax = plt.subplots()

        ax.imshow(img, cmap='gray')
        box_wtc = patches.Rectangle(rect_wtc, 
                                rect_w,
                                rect_h,
                                fill=False, edgecolor='r')

        box = patches.Rectangle(rect, rect_w, rect_h,
                                fill=False, edgecolor='b')

        ax.add_patch(box_wtc)
        ax.add_patch(box)
        plt.savefig(f'girl-wtc-frame{i}.jpg')

    np.save('girlseqrects-wcrt.npy',rects_wtc)