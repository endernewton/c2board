"""## Generation of summaries."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import io
import logging
import numpy as np
import re as _re
from PIL import Image
from six import StringIO
from six.moves import range

from c2board.src.summary_pb2 import Summary, HistogramProto, SummaryMetadata
from c2board.src.tensor_pb2 import TensorProto
from c2board.x2num import make_np

_INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')

# X: some basic function
def _clean_tag(name):
    if name is not None:
        new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
        new_name = new_name.lstrip('/')  # Remove leading slashes
        if new_name != name:
            logging.info('Summary name %s is illegal; using %s instead.' % (name, new_name))
            name = new_name
    return name

# X: this is to get the scalar out and 
def scalar(name, scalar, collections=None):
    """Assume for the scalar values are NOT needed to fetch."""
    name = _clean_tag(name)
    assert np.isscalar(scalar), 'ERROR: scalar values are not to be fetched'
    scalar = np.float32(scalar)
    return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])

# X: for histogram
def histogram(name, values, bins, collections=None):
    """Outputs a `Summary` protocol buffer with a histogram."""
    name = _clean_tag(name)
    # retrieve values 
    values = make_np(values)
    hist = make_histogram(values.astype(np.float32, copy=False), bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def make_histogram(values, bins):
    """Convert values into a histogram proto using logic from histogram.cc."""
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    # X: why?
    limits = limits[1:]

    sum_sq = values.dot(values)
    # X: hmm, tensorboard will just judge based on these statistics
    return HistogramProto(min=values.min(),
                          max=values.max(),
                          num=len(values),
                          sum=values.sum(),
                          sum_squares=sum_sq,
                          bucket_limit=limits,
                          bucket=counts)

# X: the order is BGR now
def image(tag, tensor):
    """Outputs a `Summary` protocol buffer with images."""
    tag = _clean_tag(tag)
    # X: just get the output of the images tiled if needed
    tensor = make_np(tensor, 'IMG')
    # assuming the values are within [0,255] now
    image = make_image(tensor.astype(np.uint8, copy=False))
    return Summary(value=[Summary.Value(tag=tag, image=image)])

def make_image(tensor):
    """Convert an numpy representation image to Image protobuf"""
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)


def audio(tag, tensor, sample_rate=44100):
    raise NotImplementedError

def pr_curve(tag, labels, predictions, num_thresholds=127, weights=None):
    raise NotImplementedError

# X: support text
def text(tag, text):
    PluginData = [SummaryMetadata.PluginData(plugin_name='text')]
    smd = SummaryMetadata(plugin_data=PluginData)
    tensor = TensorProto(dtype='DT_STRING',
                         string_val=[text.encode(encoding='utf_8')],
                         tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
    return Summary(value=[Summary.Value(node_name=tag, metadata=smd, tensor=tensor)])

