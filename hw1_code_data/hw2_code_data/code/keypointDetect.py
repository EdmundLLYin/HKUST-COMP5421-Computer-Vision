import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    # DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_pyramid = np.empty([gaussian_pyramid.shape[0], gaussian_pyramid.shape[1],gaussian_pyramid.shape[2]-1])
    DoG_pyramid[:,:,:gaussian_pyramid.shape[2]-1] = gaussian_pyramid[:,:,1:gaussian_pyramid.shape[2]]-gaussian_pyramid[:,:,:gaussian_pyramid.shape[2]-1]
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...
    # Compute principal curvature here
    dx = cv2.Sobel(DoG_pyramid[:,:],-1,1,0)
    dy = cv2.Sobel(DoG_pyramid[:,:],-1,0,1)
    dxx = cv2.Sobel(dx, -1, 1, 0)
    dxy = cv2.Sobel(dx, -1, 0, 1)
    dyx = cv2.Sobel(dy, -1, 1, 0)
    dyy = cv2.Sobel(dy, -1, 0, 1)
    
    detH = dxx * dyy - dxy * dyx
    traceH = dxx + dyy
    detH[detH == 0] = 1e-100#np.finfo(float).tiny
    principal_curvature = traceH * traceH / detH

    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    # locsDoG = None
    # ##############
    # #  TO DO ...
    # # Compute locsDoG here

    enableEdgeSuppression = True

    imH, imW, levels = DoG_pyramid.shape

    side_neighbors = [DoG_pyramid[:imH-2, :imW-2], DoG_pyramid[:imH-2, 1:imW-1],
                    DoG_pyramid[:imH-2, 2:], DoG_pyramid[1:imH-1, :imW-2],
                    DoG_pyramid[1:imH-1, 2:], DoG_pyramid[2:, :imW-2],
                    DoG_pyramid[2:, 1:imW-1], DoG_pyramid[2:, 2:]]

    layer_data = DoG_pyramid[1:imH-1, 1:imW-1]

    top_neighbor = np.empty((imH-2,imW-2,levels))
    top_neighbor[:,:,0:levels-1] = DoG_pyramid[1:imH-1, 1:imW-1,1:levels]

    bottom_neighbor = np.empty((imH-2,imW-2,levels))
    bottom_neighbor[:,:,1:levels] = DoG_pyramid[1:imH-1, 1:imW-1,0:levels-1]

    neighbors_for_bottom = np.array(side_neighbors + [top_neighbor])
    neighbors_for_top = np.array(side_neighbors + [bottom_neighbor])
    neighbors_for_middle = np.array(side_neighbors + [bottom_neighbor] + [top_neighbor])

    extremaMask = np.empty((imH-2,imW-2,levels))
    extremaMask[:,:,0] = (layer_data[:,:,0] > np.max(neighbors_for_bottom[:,:,:,0], axis=0)) | (layer_data[:,:,0] < np.min(neighbors_for_bottom[:,:,:,0], axis=0))
    extremaMask[:,:,levels-1] = (layer_data[:,:,levels-1] > np.max(neighbors_for_top[:,:,:,levels-1], axis=0)) | (layer_data[:,:,levels-1] < np.min(neighbors_for_top[:,:,:,levels-1], axis=0))
    extremaMask[:,:,1:levels-1] = (layer_data[:,:,1:levels-1] > np.max(neighbors_for_middle[:,:,:,1:levels-1], axis=0)) | (layer_data[:,:,1:levels-1] < np.min(neighbors_for_middle[:,:,:,1:levels-1], axis=0))
    c_mask = (abs(DoG_pyramid[1:imH-1, 1:imW-1,:]) > th_contrast)
    r_mask = (abs(principal_curvature[1:imH-1, 1:imW-1,:]) < th_r)

    if enableEdgeSuppression:
        extremaMask = (extremaMask == True) & (c_mask == True) & (r_mask == True)
    else:
        extremaMask = (extremaMask == True) & (c_mask == True)

    locations = np.where(extremaMask == True)
    x, y, l = locations[0]+1, locations[1]+1, locations[2]

    locsDoG = np.swapaxes(np.array([x,y,l]),0,1)

    return locsDoG

    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here

    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast, th_r)

    return locsDoG, gauss_pyramid







if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    show_im = cv2.resize(im, (2*im.shape[1], 2*im.shape[0]))
    for point in list(locsDoG):
        cv2.circle(show_im, (2*point[1], 2*point[0]), 2, (0, 255, 0), -1)
    cv2.imshow('output image', show_im)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()
