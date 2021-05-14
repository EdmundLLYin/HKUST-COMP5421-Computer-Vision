import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
	p = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

	th = 0.001

	x_min, y_min, x_max, y_max = 0, 0, It.shape[1]-1, It.shape[0]-1
	delta_p = np.array([3211])

	spline_It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

	while np.sum(delta_p ** 2) >= th:
		current_x_vector = np.arange(x_min, x_max+1e-9)
		current_y_vector = np.arange(y_min, y_max+1e-9)
		current_x_stack, current_y_stack = np.meshgrid(current_x_vector, current_y_vector)

		transformed_x_stack = p[0]*current_x_stack + p[1]*current_y_stack + p[2]
		transformed_y_stack = p[3]*current_x_stack + p[4]*current_y_stack + p[5]

		valid_position = (transformed_x_stack > 0) & \
			 	(transformed_x_stack < It.shape[1]) & \
				(transformed_y_stack > 0) &  \
				(transformed_y_stack < It.shape[0])

		transformed_x_stack = transformed_x_stack[valid_position]
		transformed_y_stack = transformed_y_stack[valid_position]

		current_x_stack = current_x_stack[valid_position].flatten()
		current_y_stack = current_y_stack[valid_position].flatten()

		transformed_image_intensity = spline_It1.ev(transformed_y_stack, transformed_x_stack)
		transformed_image_gradient_x = spline_It1.ev(transformed_y_stack, transformed_x_stack, dx=0, dy=1).flatten()
		transformed_image_gradient_y = spline_It1.ev(transformed_y_stack, transformed_x_stack, dx=1, dy=0).flatten()

		A = np.zeros((transformed_image_gradient_x.shape[0], 6))
		FX, FY = current_x_stack.flatten(), current_y_stack.flatten()
		A[:, 0] = np.multiply(transformed_image_gradient_x, FX)
		A[:, 1] = np.multiply(transformed_image_gradient_x, FY)
		A[:, 2] = transformed_image_gradient_x
		A[:, 3] = np.multiply(transformed_image_gradient_y, FX)
		A[:, 4] = np.multiply(transformed_image_gradient_y, FY)
		A[:, 5] = transformed_image_gradient_y

		b = It[valid_position].flatten() - transformed_image_intensity.flatten()
		inv_H = np.linalg.inv(np.dot(np.transpose(A), A))
		b_ = np.dot(np.transpose(A),b)
		delta_p = np.dot(inv_H, b_)

		p += delta_p.flatten()

	M = np.reshape(p, (2, 3))
	return M
