import numpy as np
import cv2
from BRIEF import briefLite, briefMatch
import math

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...

    A = np.zeros((p1.shape[1]*2,9))
    A[0:p1.shape[1]*2:2,0] = p2[0, :]
    A[0:p1.shape[1]*2:2,1] = p2[1, :]
    A[0:p1.shape[1]*2:2,2] = 1
    A[0:p1.shape[1]*2:2,6] = -p2[0, :] * p1[0, :]
    A[0:p1.shape[1]*2:2,7] = -p2[1, :] * p1[0, :]
    A[0:p1.shape[1]*2:2,8] = -p1[0, :]
    A[1:p1.shape[1]*2:2,3] = -p2[0, :]
    A[1:p1.shape[1]*2:2,4] = -p2[1, :]
    A[1:p1.shape[1]*2:2,5] = -1
    A[1:p1.shape[1]*2:2,6] = p2[0, :] * p1[1, :]
    A[1:p1.shape[1]*2:2,7] = p2[1, :] * p1[1, :]
    A[1:p1.shape[1]*2:2,8] = p1[1, :]
    u, s, vh = np.linalg.svd(A)
    H2to1 = vh[-1, :].reshape((3, 3))

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    
    p1 = np.empty_like(matches)
    p2 = np.empty_like(matches)
    p1 = locs1[matches[:,0],:2]
    p2 = locs2[matches[:,1],:2]
    p1_homography = np.transpose(np.hstack((p1, np.ones((p1.shape[0], 1)))))
    p2_homography = np.transpose(np.hstack((p2, np.ones((p2.shape[0], 1)))))

    max_inliers = 0
    bestH = np.zeros((3,3))

    while num_iter > 0:
        index = (np.random.random((1, 4)) * matches.shape[0]).astype(np.int)
        p1_test = np.transpose(p1[index][0, :, :])
        p2_test = np.transpose(p2[index][0, :, :])
        H2to1 = computeH(p1_test, p2_test)

        p2_transformed = np.dot(H2to1, p2_homography)
        p2_transformed /= p2_transformed[2, :]
        
        distance = np.sqrt(np.sum(np.square(p1_homography - p2_transformed), axis=0))
        num_inliers = distance[distance <= tol].size
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            bestH = H2to1

        num_iter -= 1

    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

