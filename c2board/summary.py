from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import numpy as np
import re as _re
from PIL import Image

from c2board.src.summary_pb2 import Summary, HistogramProto, SummaryMetadata
from c2board.src.tensor_pb2 import TensorProto
from c2board.x2num import make_np

_INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')

def clean_tag(name):
    '''Clean up the name.'''
    if name is not None:
        new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
        new_name = new_name.lstrip('/')  # Remove leading slashes
        if new_name != name:
            print('Summary name %s is illegal; using %s instead.' % (name, new_name))
            name = new_name
    return name

def scalar(name, scalar):
    '''Assume for the scalar values are NOT needed to fetch.'''
    assert np.isscalar(scalar), 'ERROR: scalar values are not to be fetched'
    scalar = np.float32(scalar)
    return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])

def histogram(name, values, bins):
    '''Outputs a `Summary` protocol buffer with a histogram.'''
    values = make_np(values)
    return self.histogram_with_values(name, values, bins)

def histogram_with_values(name, values, bins):
    hist = make_histogram(values.astype(np.float32, copy=False), bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])

def make_histogram(values, bins):
    '''Convert values into a histogram proto.'''
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    counts = counts.astype(np.float64) / np.float64(values.size)
    limits = limits[1:]

    sum_sq = values.dot(values)
    return HistogramProto(min=values.min(),
                          max=values.max(),
                          num=len(values),
                          sum=values.sum(),
                          sum_squares=sum_sq,
                          bucket_limit=limits,
                          bucket=counts)

def image(tag, 
        tensor, 
        rescale=0.6,
        mean_pixs=(122.7717, 115.9465, 102.9801)):
    '''Outputs a `Summary` protocol buffer with images.'''
    tensor = make_np(tensor, 'IMG') + mean_pixs
    # Assuming the values are within [0,255] now
    image = make_image(tensor.astype(np.uint8, copy=False), rescale=rescale)
    return Summary(value=[Summary.Value(tag=tag, image=image)])

def make_image(tensor, rescale):
    '''Convert an numpy representation image to Image protobuf.'''
    height, width, channel = tensor.shape
    theight = int(height * rescale)
    twidth = int(width * rescale)
    image = Image.fromarray(tensor)
    image = image.resize((twidth, theight), Image.ANTIALIAS)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=theight,
                         width=twidth,
                         colorspace=channel,
                         encoded_image_string=image_string)

def text(tag, text):
    '''Outputs a `Summary` protocol buffer with text.'''
    PluginData = [SummaryMetadata.PluginData(plugin_name='text')]
    smd = SummaryMetadata(plugin_data=PluginData)
    tensor = TensorProto(dtype='DT_STRING',
                 string_val=[text.encode(encoding='utf_8')],
                 tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
    return Summary(value=[Summary.Value(node_name=tag, metadata=smd, tensor=tensor)])

def audio(tag, tensor, sample_rate=44100):
    raise NotImplementedError

def pr_curve(tag, labels, predictions, num_thresholds=127, weights=None):
    raise NotImplementedError

