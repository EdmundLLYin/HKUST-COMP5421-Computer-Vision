import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def replace_image(mask_image, source_image, target_image):
    mask = np.count_nonzero(mask_image, axis=2)
    mask = np.array(np.nonzero(mask))
    target_image[mask[0,:],mask[1,:]] = source_image[mask[0,:],mask[1,:]]
    return target_image

def get_blend_weights(im1, im2, h1, h2, out_shape):

    mask1 = np.ones((im1.shape[0], im1.shape[1]))
    mask1[0, :], mask1[-1, :], mask1[:, 0], mask1[:, -1] = 0, 0, 0, 0
    mask1 = distance_transform_edt(mask1)
    mask1 /= np.max(mask1)
    weight1 = cv2.warpPerspective(mask1, h1, out_shape)

    mask2 = np.ones((im2.shape[0], im2.shape[1]))
    mask2[0, :], mask2[-1, :], mask2[:, 0], mask2[:, -1] = 0, 0, 0, 0
    mask2 = distance_transform_edt(mask2)
    mask2 /= np.max(mask2)
    weight2 = cv2.warpPerspective(mask2, h2, out_shape)

    sum_weight = weight1 + weight2
    sum_weight[sum_weight == 0] = 1.0
    weight1 /= sum_weight
    weight2 /= sum_weight
    weight1 = np.stack((weight1, weight1, weight1), axis=2)
    weight2 = np.stack((weight2, weight2, weight2), axis=2)

    return weight1, weight2

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...

    im1h, im1w, _ = im1.shape
    im2h, im2w, _ = im2.shape

    source_points = np.array([  np.array([0, 0, 1]), np.array([0, im2h-1, 1]),
                                np.array([im2w-1, 0, 1]), np.array([im2w-1, im2h-1, 1])])
    target_points = np.dot(H2to1, np.transpose(source_points))
    target_points /= target_points[2, :]
    target_width, target_height = int(np.round(np.max(target_points[0, :]))), int(np.round(np.max(target_points[1, :])))
    warp_im2 = cv2.warpPerspective(im2, H2to1, (target_width, target_height))
    cv2.imwrite('../results/6_1.jpg', warp_im2)
    

    weight1, weight2 = get_blend_weights(im1, im2, np.float32([[1,0,0],[0,1,0],[0,0,1]]), H2to1, (target_width, target_height))

    pano_im = np.zeros((max(im1h, target_height), max(im1w, target_width), 3), im1.dtype)
    pano_im[:im1h, :im1w, :] = im1
    cv2.imwrite('../results/6_1_1.jpg', pano_im)
    merge_area = pano_im * weight1 + warp_im2 * weight2
    
    replace_image(warp_im2, warp_im2, pano_im)
    replace_image(pano_im * warp_im2, merge_area, pano_im)

    return pano_im

def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...

    im1h, im1w, _ = im1.shape
    im2h, im2w, _ = im2.shape

    source_points = np.array([  np.array([0, 0, 1]), np.array([0, im2h - 1, 1]),
                                np.array([im2w - 1, 0, 1]), np.array([im2w - 1, im2h - 1, 1])])
    target_points = np.dot(H2to1, np.transpose(source_points))
    target_points /= target_points[2, :]
    min_width, min_height = int(np.round(np.min(target_points[0, :]))), int(np.round(np.min(target_points[1, :])))
    target_width, target_height = max(-min_width, 0) + int(np.round(np.max(target_points[0, :]))), max(-min_height, 0) + int(np.round(np.max(target_points[1, :])))
    out_size = (target_width, target_height)
    M = np.float32([[1, 0, max(-min_width, 0)], [0, 1, max(-min_height, 0)], [0, 0, 1]])
    
    warp_im1 = cv2.warpPerspective(im1, M, out_size)
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M,H2to1), out_size)
    cv2.imwrite('../results/q6_2_1.jpg', warp_im1)
    cv2.imwrite('../results/q6_2_2.jpg', warp_im2)

    weight1, weight2 = get_blend_weights(im1, im2, M, np.matmul(M,H2to1), out_size)

    pano_im = np.zeros((out_size[1], out_size[0], 3), im1.dtype)
    replace_image(warp_im1, warp_im1, pano_im)
    replace_image(warp_im2, warp_im2, pano_im)
    replace_image(pano_im * warp_im2, warp_im1 * weight1 + warp_im2 * weight2, pano_im)

    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)

    return pano_im

def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    np.save('../results/q6_1.npy', H2to1)
    #im3 = imageStitching(im1, im2, H2to1)
    im3 = imageStitching_noClip(im1, im2, H2to1)
    return im3


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    
    pano_im = generatePanorama(im1, im2)

    cv2.imwrite('../results/q6_3.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()