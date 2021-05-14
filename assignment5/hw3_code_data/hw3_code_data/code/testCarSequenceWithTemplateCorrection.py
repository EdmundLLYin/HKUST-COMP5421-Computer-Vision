import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2
import LucasKanade

# write your script here, we recommend the above libraries for making your animation

if __name__ == '__main__':
    target_path = '../result/'

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    video = np.load('../data/carseq.npy')

    #print(video.shape)
    template_frame = video[:, :, 0]

    rects = []
    template_rect = np.array([59, 116, 145, 151])
    rects.append(template_rect)

    pre_p = None
    update_th = 0.05

    ori_rects = np.load('carseqrects.npy')
    
    for i in range(1, video.shape[2]):
        next_frame = video[:, :, i]

        p = LucasKanade.LucasKanade(template_frame, next_frame, template_rect)

        rect = [template_rect[0]+p[0], template_rect[1]+p[1], template_rect[2]+p[0], template_rect[3]+p[1]]
        rects.append(rect)
        
        if pre_p is None:
            pre_p = np.copy(p)

        if np.sqrt(np.sum((p-pre_p)**2)) < update_th:
            template_frame = next_frame
            template_rect = rect

        img = np.zeros((next_frame.shape[0], next_frame.shape[1], 3))
        img[:, :, 0] = next_frame
        img[:, :, 1] = next_frame
        img[:, :, 2] = next_frame
        cv2.rectangle(img, (int(round(ori_rects[i][0])), int(round(ori_rects[i][1]))),
                      (int(round(ori_rects[i][2])), int(round(ori_rects[i][3]))),
                      color=(0,255,0), thickness=2)
        cv2.rectangle(img, (int(round(rect[0])), int(round(rect[1]))), (int(round(rect[2])), int(round(rect[3]))),
                      color=(0, 255, 255), thickness=2)
        cv2.imshow('image', img)
        cv2.waitKey(1)

        if i in [1, 100, 200, 300, 400]:
            cv2.imwrite(os.path.join(target_path, 'q1_4_{}.jpg'.format(i)), img*255)

    rects = np.array(rects)
    #print(rect_list)
    np.save('carseqrects-wcrt.npy', rects)
