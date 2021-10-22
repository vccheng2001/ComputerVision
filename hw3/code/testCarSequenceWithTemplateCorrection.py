import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import *
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=0.01, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

# (h=240,w=320,indices=415)
# first frame: imshow(frames[:,:,0])
seq = np.load("../data/carseq.npy")


# cv2.imshow('frame0', seq[:,:,0])
# cv2.waitKey()

# x1,y1,x2,y2
rect = [59, 116, 145, 151]
rect_wtc = [59, 116, 145, 151]
h, w, num_frames = seq.shape
print('hwf', h,w,num_frames) # 415 frames 
rects = []
rects_wtc = []

prev_template = None
for i in range(1, num_frames-1):
    print(f'Frame {i}')

    # for regular LK 
    template = seq[:,:,i-i] * 255
    img = seq[:,:,i] * 255 # image at t+1


    # for LK WTC: only update if not using prev template
    if prev_template is None:
        update_template = True
        template_wtc = seq[:,:,i-1] * 255
    else:
        update_template = False
        template_wtc = prev_template

    # run LK on each frame pair
    p  = LucasKanade(template, img, rect, threshold, num_iters, p0=np.zeros(2))
    p_wtc, prev_template = LucasKanadeWTC(template_wtc, img, rect_wtc, threshold, num_iters, p0=np.zeros(2), update_template=update_template)

    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    rect_wtc[0] += p_wtc[0]
    rect_wtc[1] += p_wtc[1]
    rect_wtc[2] += p_wtc[0]
    rect_wtc[3] += p_wtc[1]

    rects.append(rect)
    rects_wtc.append(rect_wtc)





    # if i == 1 or i % 100 == 0:
    #     l,w = rect[2]-rect[0], rect[3]-rect[1]
    #     plt.figure()
    #     plt.imshow(seq[:,:,i],cmap='gray')
    #     bbox0 = patches.Rectangle((int(rects[i-1][0]), int(rects[i-1][1])), l,w,
    #                              fill=False, edgecolor='red', linewidth=2)
    #     plt.gca().add_patch(bbox0)
    #     bbox1 = patches.Rectangle((int(rects_wtc[i-1][0]), int(rects_wtc[i-1][1])), l,w,
    #                              fill=False, edgecolor='blue', linewidth=2)
    #     plt.gca().add_patch(bbox1)
    #     plt.title('Frame %d' % i)
    #     plt.savefig(f'car_wtc_frame{i}.jpg')

    if i == 1 or i % 100 == 0:
        x1,y1,x2,y2 = map(int, rect)
        x1_wtc,y1_wtc,x2_wtc,y2_wtc= map(int, rect_wtc)

        # img_with_rect = cv2.rectangle(img=img.copy(),
        #                              pt1=(x1,y1), 
        #                              pt2=(x2,y2), 
        #                              color=(255,0,0), 
        #                              thickness=1)


        # img_with_rect_wtc = cv2.rectangle(img=img_with_rect.copy(),
        #                              pt1=(x1_wtc,y1_wtc), 
        #                              pt2=(x2_wtc,y2_wtc), 
        #                              color=(0,255,0), 
        #                              thickness=1)

        img_with_rect_wtc = cv2.rectangle(img=img.copy(),
                                     pt1=(x1_wtc,y1_wtc), 
                                     pt2=(x2_wtc,y2_wtc), 
                                     color=(0,255,0), 
                                     thickness=1)

        # cv2.imshow('img_with_rect',img)
        cv2.imwrite(f'car-wcrt_frame{i}.jpg', img_with_rect_wtc)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()


# save all rects as Nx4 matrix
out = 'carseqrects-wcrt.npy'
with open(out, 'wb') as f:
    print('Saving.....')
    np.save(f, rects_wtc)


