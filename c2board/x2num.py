# X: this should be the script to convert any objects to numpy arrays
# X: do not do any image processing here
import numpy as np

from caffe2.python import workspace


def make_np(x, modality=None):
    # if already numpy, return
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    assert isinstance(x, str), 'ERROR: should pass name of the blob here'
    # X: everything else, just fetch the blob given the name
    return caffe2_num(x, modality)

def caffe2_num(x, modality=None):
    # get the string and output the caffe2 numpy array
    x = workspace.FetchBlob(x)
    if modality == 'IMG':
        x = _prepare_image(x)
    return x

# X: to put images in the grid
def make_grid(I, ncols=4):
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

# X: tensorflow is NHWC
# X: caffe2 is NCHW
def _prepare_image(I):
    assert isinstance(I, np.ndarray), 'ERROR: should pass numpy array here'
    assert I.ndim == 2 or I.ndim == 3 or I.ndim == 4, \
        'ERROR: image should have dimensions of 2, 3, or 4'
    if I.ndim == 4:  # NCHW
        if I.shape[1] == 1:  # N1HW
            I = np.concatenate((I, I, I), 1)  # N3HW
        assert I.shape[1] == 3
        I = make_grid(I)  # 3xHxW
    if I.ndim == 3 and I.shape[0] == 1:  # 1xHxW
        I = np.concatenate((I, I, I), 0)  # 3xHxW
    if I.ndim == 2:  # HxW
        I = np.expand_dims(I, 0)  # 1xHxW
        I = np.concatenate((I, I, I), 0)  # 3xHxW
    # X: so that finally it is HWC
    I = I.transpose(1, 2, 0)

    return I
