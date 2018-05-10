from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import numpy as np
import re as _re
from random import shuffle

import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from c2board.src.summary_pb2 import Summary, HistogramProto, SummaryMetadata
from c2board.src.tensor_pb2 import TensorProto
from c2board.x2num import make_np

_INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)
FONT = ImageFont.load_default()

def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font, color='black', color_text='black', thickness=2):
  draw = ImageDraw.Draw(image)
  (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  if display_str:
      text_bottom = bottom
      # Reverse list and print from bottom to top.
      text_width, text_height = font.getsize(display_str)
      margin = np.ceil(0.05 * text_height)
      draw.rectangle(
          [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                            text_bottom)],
          fill=color)
      draw.text(
          (left + margin, text_bottom - text_height - margin),
          display_str,
          fill=color_text,
          font=font)
  return image

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
    tensor = make_np(tensor, 'IMG')
    if isinstance(tensor, list):
        res = []
        for i, t in enumerate(tensor):
            t = t + mean_pixs
            image = make_image(t.astype(np.uint8, copy=False), rescale=rescale)
            res.append(Summary(value=[Summary.Value(tag=tag + ('_%d' % i), image=image)]))
        return res
    else:
        tensor = tensor + mean_pixs
        # Assuming the values are within [0,255] now
        image = make_image(tensor.astype(np.uint8, copy=False), rescale=rescale)
        return Summary(value=[Summary.Value(tag=tag, image=image)])

def image_boxes(tag, 
                tensor_image,
                tensor_boxes,
                rescale=0.8,
                mean_pixs=(122.7717, 115.9465, 102.9801)):
    '''Outputs a `Summary` protocol buffer with images.'''
    tensor_image = make_np(tensor_image, 'IMG')
    tensor_boxes = make_np(tensor_boxes)
    if isinstance(tensor_image, list):
        res = []
        for i, t in enumerate(tensor_image):
            t = t + mean_pixs
            rois = tensor_boxes[tensor_boxes[:,0]==i, 1:5]
            image = make_image(t.astype(np.uint8, copy=False), 
                                        rescale=rescale,
                                        rois=rois)
            res.append(Summary(value=[Summary.Value(tag=tag + ('_%d' % i), image=image)]))
        return res
    else:
        tensor_image = tensor_image + mean_pixs
        # Assuming the values are within [0,255] now
        rois = tensor_boxes[:, 1:5]
        image = make_image(tensor_image.astype(np.uint8, copy=False), 
                           rescale=rescale,
                           rois=rois)
        return Summary(value=[Summary.Value(tag=tag, image=image)])

def draw_boxes(disp_image, gt_boxes):
    num_boxes = gt_boxes.shape[0]
    list_gt = range(num_boxes)
    shuffle(list_gt)
    for i in list_gt:
        disp_image = _draw_single_box(disp_image, 
                                    gt_boxes[i, 0],
                                    gt_boxes[i, 1],
                                    gt_boxes[i, 2],
                                    gt_boxes[i, 3],
                                    None,
                                    FONT,
                                    color='Red')
    return disp_image

def make_image(tensor, rescale, rois=None):
    '''Convert an numpy representation image to Image protobuf.'''
    height, width, channel = tensor.shape
    theight = int(height * rescale)
    twidth = int(width * rescale)
    image = Image.fromarray(tensor)
    if rois is not None:
        image = draw_boxes(image, rois)
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

