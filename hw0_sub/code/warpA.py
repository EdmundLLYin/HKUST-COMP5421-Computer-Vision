import numpy as np
from numpy.linalg import inv

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""

    A = inv(A)
    output = np.empty_like(im)

    for j in range(output_shape[0]):
        for i in range(output_shape[1]):
            x, y, z = A.dot(np.array([j,i,1]))
            x = round(x)
            y = round(y)
            if x >= 0 and y >=0 and x < output_shape[0] and y < output_shape[1]:
                output[j,i] = im[round(x),round(y)]
            else:
                output[j,i] = 0

    return output
