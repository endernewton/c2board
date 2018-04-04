from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from c2board.writer import SummaryWriter
from caffe2.python.models import bvlc_alexnet


with SummaryWriter(tag='bvlc_alexnet') as writer:
    writer.write_graph([bvlc_alexnet.predict_net])
