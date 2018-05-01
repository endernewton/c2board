from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from caffe2.python import workspace

def make_nps(xs):
    assert isinstance(xs, list), 'ERROR: should pass list of names of the blobs'
    return workspace.FetchBlobs(xs)

def make_np(x, modality=None):
    '''Make any blob to numpy arrays.'''
    if isinstance(x, np.ndarray):
        return x
    assert isinstance(x, unicode), 'ERROR: should pass name of the blob here'
    return caffe2_num(x, modality)

def caffe2_num(x, modality=None):
    '''Fetch blobs for caffe.'''
    x = workspace.FetchBlob(x)
    if modality == 'IMG':
        x = _prepare_image(x)
    return x

def make_grid(I, ncols=4):
    '''Make a grid view out of all the images.'''
    assert isinstance(I, np.ndarray), 'ERROR: should pass numpy array here'
    assert I.ndim == 4 and I.shape[1] == 3, \
        'ERROR: should have dimension for N to put them in grid'
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((3, H * nrows, W * ncols))
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * H:(y + 1) * H, x * W:(x + 1) * W] = I[i]
            i = i + 1
    return canvas

def _prepare_image(I):
    '''Prepare images, such as changing the format, etc.'''
    assert isinstance(I, np.ndarray), 'ERROR: should pass numpy array here'
    assert I.ndim == 2 or I.ndim == 3 or I.ndim == 4, \
        'ERROR: image should have dimensions of 2, 3, or 4'
    if I.ndim == 4:  # NCHW
        if I.shape[1] == 1:  # N1HW
            I = np.concatenate((I, I, I), 1)  # N3HW
        assert I.shape[1] == 3
        I = np.split(I, I.shape[0], 0)  # 1x3xHxW
        I = [np.squeeze(i) for i in I]  # 3xHxW
        I = [i.transpose(1,2,0) for i in I]
        I = [i[...,::-1] for i in I]
    elif I.ndim == 3 and I.shape[0] == 1:  # 1xHxW
        I = np.concatenate((I, I, I), 0)  # 3xHxW
        # Finally, HWC
        I = I.transpose(1, 2, 0)
        # RGB instead of BGR
        I = I[...,::-1]
    elif I.ndim == 2:  # HxW
        I = np.expand_dims(I, 0)  # 1xHxW
        I = np.concatenate((I, I, I), 0)  # 3xHxW
        # Finally, HWC
        I = I.transpose(1, 2, 0)
        # RGB instead of BGR
        I = I[...,::-1]

    return I
