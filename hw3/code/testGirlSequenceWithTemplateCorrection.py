import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import *


parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

h, w, num_frames = seq.shape
rects = []

for i in range(num_frames-1):
    print(f'Frame {i}')


#    only update if not using prev template
    if not prev_template:
        # track optical flow from frame to frame
        template = seq[:,:,i] * 255
    else:
        template = prev_template

    img = seq[:,:,i+1] * 255 # image at t+1

    # run LK on each frame pair
    p, prev_template = LucasKanadeWTC(template, img, rect, threshold, num_iters, p0=np.zeros(2))
    
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    rects.append(rect)


    if i % 20 == 0:
        x1,y1,x2,y2 = map(int, rect)

        img_with_rect= cv2.rectangle(img=img.copy(), 
                                     pt1=(x1,y1), 
                                     pt2=(x2,y2), 
                                     color=(255,0,0), 
                                     thickness=3)
        # cv2.imshow('img_with_rect',img_with_rect)
        cv2.imwrite(f'girl-wcrt_frame{i}.jpg', img_with_rect)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


# save all rects as Nx4 matrix
out = 'girlseqrects-wcrt.npy'
with open(out, 'wb') as f:
    print('Saving.....')
    np.save(f, rects)
