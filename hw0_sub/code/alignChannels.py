import numpy as np
from numpy import linalg as LA
import time

loop = 3
alignment_range = 30

def get_target_shift(red, green, blue):

    dim1 = len(red[0])

    rg_sim = []
    rb_sim = []

    red_norm = red[:,alignment_range:dim1-alignment_range] / LA.norm(red[:,alignment_range:dim1-alignment_range])

    for offset in range(-alignment_range, alignment_range+1, 1):

      dot_rg = np.multiply(red_norm, green[:,alignment_range+offset:dim1-alignment_range+offset] / LA.norm(green[:,alignment_range+offset:dim1-alignment_range+offset]))
      dot_rb = np.multiply(red_norm, blue[:,alignment_range+offset:dim1-alignment_range+offset] / LA.norm(blue[:,alignment_range+offset:dim1-alignment_range+offset]))
      
      rg_sim.append(np.sum(dot_rg))
      rb_sim.append(np.sum(dot_rb))

    return -(np.argmax(rg_sim)-alignment_range), -(np.argmax(rb_sim)-alignment_range)

def shift_image(red, green, blue, gx, gy, bx, by):

    dim0 = len(red)
    dim1 = len(red[0])

    red_pad = np.pad(red, (alignment_range, alignment_range), 'constant', constant_values=(0, 0))
    green_pad = np.pad(green, (alignment_range, alignment_range), 'constant', constant_values=(0, 0))
    blue_pad = np.pad(blue, (alignment_range, alignment_range), 'constant', constant_values=(0, 0))

    red_pad[alignment_range:dim0+alignment_range,alignment_range:dim1+alignment_range] = red_pad[alignment_range:dim0+alignment_range,alignment_range:dim1+alignment_range]
    green_pad[alignment_range+gx:dim0+alignment_range+gx,gy+alignment_range:gy+dim1+alignment_range] = green_pad[alignment_range:dim0+alignment_range,alignment_range:dim1+alignment_range]
    blue_pad[alignment_range+bx:dim0+alignment_range+bx,by+alignment_range:by+dim1+alignment_range] = blue_pad[alignment_range:dim0+alignment_range,alignment_range:dim1+alignment_range]

    red[0:dim0, 0:dim1] = red_pad[alignment_range:dim0+alignment_range,alignment_range:dim1+alignment_range] 
    green[0:dim0, 0:dim1] = green_pad[alignment_range:dim0+alignment_range,alignment_range:dim1+alignment_range]
    blue[0:dim0, 0:dim1] = blue_pad[alignment_range:dim0+alignment_range,alignment_range:dim1+alignment_range]

    return red, green, blue


def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""

    for _ in range(loop):
  
      gy, by = get_target_shift(red, green, blue)
      red, green, blue = shift_image(red, green, blue, 0, gy, 0, by)

      gx, bx = get_target_shift(np.swapaxes(red,0,1), np.swapaxes(green,0,1), np.swapaxes(blue,0,1))
      red, green, blue = shift_image(red, green, blue, gx, 0, bx, 0)

    rgb = []
    rgb.append(red)
    rgb.append(green)
    rgb.append(blue)

    rgb = np.array(rgb)
    rgb = np.moveaxis(rgb, 0, -1)

    return np.array(rgb)
