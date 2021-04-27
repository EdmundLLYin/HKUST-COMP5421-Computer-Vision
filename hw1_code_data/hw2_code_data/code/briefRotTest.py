import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
from BRIEF import briefLite, briefMatch, plotMatches
import matplotlib.pyplot as plt

# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def check_correct_matches(image, angle, matches, locs1, locs2):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    locs1[:,2] = 1
    rotated = rot_mat.dot(locs1.T).T
    locs2 = locs2[:,:2]
    #correct_mask = [(np.round(locs2[matches[:,1],0]) == np.round(rotated[matches[:,0],0])) & (np.round(locs2[matches[:,1],1]) == np.round(rotated[matches[:,0],1]))]
    correct_mask = [(np.isclose(locs2[matches[:,1],0], rotated[matches[:,0],0], atol = 1.0)) & (np.isclose(locs2[matches[:,1],1], rotated[matches[:,0],1], atol = 1.0))]
    print("Accuracies at angle", angle, ":", correct_mask[0].sum()/ len(correct_mask[0]) * 100, "%")
    return correct_mask[0].sum(), len(correct_mask[0])

if __name__ == '__main__':
    # test briefLite
    im = cv2.imread('../data/model_chickenbroth.jpg')
    locs, desc = briefLite(im)  
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locs[:,0], locs[:,1], 'r.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    # test matches for rotation
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im1 = cv2.copyMakeBorder(im1, 30, 30, 30, 30, cv2.BORDER_CONSTANT) 
    record = []
    for angle in range(-180,181,10):
        im2 = rotate_image(im1, angle)
        locs1, desc1 = briefLite(im1)
        locs2, desc2 = briefLite(im2)
        matches = briefMatch(desc1, desc2)
        correct, total = check_correct_matches(im1, angle, matches, locs1, locs2)
        plotMatches(im1,im2,matches,locs1,locs2)
        record.append(correct)
    plt.bar(range(-180,181,10), record, 5)
    plt.xlabel("Rotation angle")
    plt.ylabel("Number of correct matches")
    plt.show()
