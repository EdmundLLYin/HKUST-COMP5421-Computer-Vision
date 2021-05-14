import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def InverseCompositionAffine(It, It1):
    # Input:
    # 	It: template image
    # 	It1: Current image

    #  Output:
    # 	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()

    th = 0.001

    x_min, y_min, x_max, y_max = 0, 0, It.shape[1] - 1, It.shape[0] - 1
    delta_p = np.array([3211])

    spline_It1 = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It1)
    spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    x_vector = np.arange(x_min, x_max + 0.5)
    y_vector = np.arange(y_min, y_max + 0.5)
    x_stack, y_stack = np.meshgrid(x_vector, y_vector)

    image_gradient_x = spline_It.ev(y_stack, x_stack, dx=0, dy=1).flatten()
    image_gradient_y = spline_It.ev(y_stack, x_stack, dx=1, dy=0).flatten()

    N = image_gradient_x.shape[0]
    A_ = np.zeros((N, 6))
    FX, FY = x_stack.flatten(), y_stack.flatten()
    A_[:, 0] = np.multiply(image_gradient_x, FX)
    A_[:, 1] = np.multiply(image_gradient_x, FY)
    A_[:, 2] = image_gradient_x
    A_[:, 3] = np.multiply(image_gradient_y, FX)
    A_[:, 4] = np.multiply(image_gradient_y, FY)
    A_[:, 5] = image_gradient_y

    while np.sum(delta_p ** 2) >= th:
        
        current_x_stack = p[0] * x_stack + p[1] * y_stack + p[2]
        current_y_stack = p[3] * x_stack + p[4] * y_stack + p[5]

        valid_position =    (current_x_stack > 0) & \
                            (current_x_stack < It1.shape[1]) & \
                            (current_y_stack > 0) & \
                            (current_y_stack < It1.shape[0])
        
        current_x_stack = current_x_stack[valid_position]
        current_y_stack = current_y_stack[valid_position]

        A_valid = A_[valid_position.flatten()]
        
        image_intensity = spline_It1.ev(current_y_stack, current_x_stack)
        
        b = image_intensity.flatten() - It[valid_position].flatten()
        inv_H = np.linalg.inv(np.dot(np.transpose(A_valid), A_valid))
        b_ = np.dot(np.transpose(A_valid), b)
        delta_p = np.dot(inv_H, b_)

        M = np.vstack((np.reshape(p, (2, 3)), np.array([[0, 0, 1]])))
        delta_M = np.vstack((np.reshape(delta_p, (2, 3)), np.array([[0, 0, 1]])))
        delta_M[0, 0] += 1
        delta_M[1, 1] += 1
        M = np.dot(M, np.linalg.inv(delta_M))

        p = M[:2, :].flatten()

    M = M[:2, :]

    return M