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

# (h=240,w=320,indices=415)
# first frame: imshow(frames[:,:,0])
seq = np.load("../data/carseq.npy")
# cv2.imshow('frame0', seq[:,:,0])
# cv2.waitKey()

rect = [59, 116, 145, 151]

h, w, num_frames = seq.shape
results = []
for i in range(num_frames-1):

    # track optical flow from frame to frame
    template = seq[:,:,i]
    img = seq[:,:,i+1] # image at t+1

    # run LK on each frame pair
    res = LucasKanade(template, img, rect, threshold, num_iters, p0=np.zeros(2))
    results.append(res)

# save all rects as Nx4 matrix
with open('carseqrects.npy', 'wb') as f:
    np.save(f, results)

