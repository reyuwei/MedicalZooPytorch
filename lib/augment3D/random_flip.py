import numpy as np


# TODO test

def random_flip(img_numpy, label=None, axis_for_flip=0, affine=None):
    axes = [0, 1, 2]

    img_numpy = flip_axis(img_numpy, axes[axis_for_flip])
    img_numpy = np.squeeze(img_numpy)

    if label is None:
        return img_numpy, label
    else:
        y = flip_axis(label, axes[axis_for_flip])
        y = np.squeeze(y)

    size = img_numpy.shape

    flip_mat = np.eye(4)
    flip_mat[axes[axis_for_flip], axes[axis_for_flip]] = -1
    flip_mat[axes[axis_for_flip], -1] = size[axes[axis_for_flip]]
    affine = affine @ flip_mat

    return img_numpy, y, affine


def flip_axis(img_numpy, axis):
    img_numpy = np.asarray(img_numpy).swapaxes(axis, 0)
    img_numpy = img_numpy[::-1, ...]
    img_numpy = img_numpy.swapaxes(0, axis)
    return img_numpy


class RandomFlip(object):
    def __init__(self):
        self.axis_for_flip = np.random.randint(0, 3)

    def __call__(self, img_numpy, label=None, affine=None):
        """
        Args:
            img_numpy (numpy): Image to be flipped.
            label (numpy): Label segmentation map to be flipped

        Returns:
            img_numpy (numpy):  flipped img.
            label (numpy): flipped Label segmentation.
        """
        return random_flip(img_numpy, label, self.axis_for_flip, affine)
