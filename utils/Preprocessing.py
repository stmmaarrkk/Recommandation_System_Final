import os
import math
import numpy as np


class Scale(object):
    '''
    Given max and min of a rectangle it computes the scale and shift values to normalize data to [0,1]
    '''

    def __init__(self):
        self.min_x = +math.inf
        self.max_x = -math.inf
        self.min_y = +math.inf
        self.max_y = -math.inf
        self.sx, self.sy = 1, 1

    def calc_scale(self, keep_ratio=True):
        self.sx = 1 / (self.max_x - self.min_x)
        self.sy = 1 / (self.max_y - self.min_y)
        if keep_ratio:
            if self.sx > self.sy:
                self.sx = self.sy
            else:
                self.sy = self.sx

    def normalize(self, data, shift=True, inPlace=True):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        if data.ndim == 1:
            data_copy[0] = (data[0] - self.min_x * shift) * self.sx
            data_copy[1] = (data[1] - self.min_y * shift) * self.sy
        elif data.ndim == 2:
            data_copy[:, 0] = (data[:, 0] - self.min_x * shift) * self.sx
            data_copy[:, 1] = (data[:, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 3:
            data_copy[:, :, 0] = (data[:, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, 1] = (data[:, :, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 4:
            data_copy[:, :, :, 0] = (data[:, :, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, :, 1] = (data[:, :, :, 1] - self.min_y * shift) * self.sy
        else:
            return False
        return data_copy

    def denormalize(self, data, shift=True, inPlace=False):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        ndim = data.ndim
        if ndim == 1:
            data_copy[0] = data[0] / self.sx + self.min_x * shift
            data_copy[1] = data[1] / self.sy + self.min_y * shift
        elif ndim == 2:
            data_copy[:, 0] = data[:, 0] / self.sx + self.min_x * shift
            data_copy[:, 1] = data[:, 1] / self.sy + self.min_y * shift
        elif ndim == 3:
            data_copy[:, :, 0] = data[:, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, 1] = data[:, :, 1] / self.sy + self.min_y * shift
        elif ndim == 4:
            data_copy[:, :, :, 0] = data[:, :, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, :, 1] = data[:, :, :, 1] / self.sy + self.min_y * shift
        else:
            return False

        return data_copy