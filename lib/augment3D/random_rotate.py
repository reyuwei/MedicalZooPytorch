import numpy as np
import scipy.ndimage as ndimage
from scipy import special

def random_rotate3D(img_numpy, min_angle, max_angle):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return: 3D rotated img
    """
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes, reshape=False), angle, axes


class RandomRotation(object):
    def __init__(self, min_angle=-10, max_angle=10):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def compute_rot_mat(self, input_arr, angle, axes):
        ndim = 3
        axes = list(axes)
        if len(axes) != 2:
            raise ValueError('axes should contain exactly two values')
        if not all([float(ax).is_integer() for ax in axes]):
            raise ValueError('axes should contain only integer values')
        if axes[0] < 0:
            axes[0] += ndim
        if axes[1] < 0:
            axes[1] += ndim
        if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
            raise ValueError('invalid rotation plane specified')
        axes.sort()
        c, s = special.cosdg(angle), special.sindg(angle)
        rot_matrix = np.array([[c, s], [-s, c]])

        img_shape = np.asarray(input_arr.shape)
        in_plane_shape = img_shape[axes]
        out_plane_shape = img_shape[axes]

        out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
        in_center = (in_plane_shape - 1) / 2
        offset = in_center - out_center

        output_shape = img_shape
        output_shape[axes] = out_plane_shape
        output_shape = tuple(output_shape)

        # np.dot(matrix, o) + offset
        rot_mat = np.eye(4)
        rot_mat[axes, -1] = offset
        for i in range(len(axes)):
            for j in range(len(axes)):
                rot_mat[axes[i], axes[j]] = rot_matrix[i,j]
        return rot_mat



    def __call__(self, img_numpy, label=None, affine=None):
        """
        Args:
            img_numpy (numpy): Image to be rotated.
            label (numpy): Label segmentation map to be rotated

        Returns:
            img_numpy (numpy): rotated img.
            label (numpy): rotated Label segmentation map.
        """
        img_numpy, angle, axes = random_rotate3D(img_numpy, self.min_angle, self.max_angle)

        rot_mat = self.compute_rot_mat(img_numpy, angle, axes)
        affine = affine @ rot_mat

        if label.any() != None:
            label = ndimage.rotate(label, angle, axes=axes, order=0, reshape=False)
            # label = random_rotate3D(label, self.min_angle, self.max_angle)
        return img_numpy, label, affine
