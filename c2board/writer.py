"""Provides an API for generating Event protocol buffers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

from c2board.src import event_pb2
from c2board.src import summary_pb2
from c2board.src import graph_pb2
from c2board.x2num import make_np

from c2board.event_file_writer import EventFileWriter
from c2board.graph_torch import graph_torch
from c2board.graph import graph
import c2board.summary as summary


class FileWriter(object):
    """Writes `Summary` protocol buffers to event files."""

    def __init__(self, 
                logdir,
                max_queue=10,
                flush_secs=120):
        """Creates a `SummaryWriter` and an event file."""
        self._event_writer = EventFileWriter(logdir, max_queue, flush_secs)
        self._closed = False

    def get_logdir(self):
        return self._event_writer.get_logdir()

    def add_summary(self, summary, global_step=None):
        """Adds a `Summary` protocol buffer to the event file."""
        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ
        event = event_pb2.Event(summary=summary)
        self._add_event(event, global_step)

    # X: this is the function to add the graph to the event
    def add_graph(self, graph):
        """Adds a `Graph` protocol buffer to the event file."""
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self._add_event(event, None)

    # X: the underlying function to add an event
    def _add_event(self, event, step):
        event.wall_time = time.time()
        if step is not None:
            event.step = int(step)
        self._event_writer.add_event(event)

    def flush(self):
        self._event_writer.flush()

    def close(self):
        self._event_writer.close()
        self._closed = True


# X: the biggest class to handle events
class SummaryWriter(object):
    """Writes `Summary` directly to event files."""
    def __init__(self, log_dir=None, tag='default'):
        if not log_dir:
            # X: just create a name for the log files
            log_dir = os.path.join('runs', tag)
        self._file_writer = FileWriter(logdir=log_dir)
        # X: next is to create bins, amazing that it can fit all the values
        v = 1E-12
        buckets = []
        neg_buckets = []
        while v < 1E20:
            buckets.append(v)
            neg_buckets.append(-v)
            v *= 1.1
        self.default_bins = neg_buckets[::-1] + [0] + buckets
        self.text_tags = []
        self.scalar_dict = {}
        self.text_dir = None

    def __append_to_scalar_dict(self, 
                                tag, 
                                scalar_value, 
                                global_step,
                                timestamp):
        """This adds an entry to the self.scalar_dict data structure with format
        {writer_id : [[timestamp, step, value], ...], ...}.
        """
        # X: it seems like it will just create a dictionary for each scalar
        if tag not in self.scalar_dict.keys():
            self.scalar_dict[tag] = []
        self.scalar_dict[tag].append([timestamp, 
                                    global_step, 
                                    float(scalar_value)])

    def add_scalar(self, tag, scalar_value, global_step=None):
        """Add scalar data to summary.
        """
        self._file_writer.add_summary(summary.scalar(tag, scalar_value), 
                                    global_step)
        self.__append_to_scalar_dict(tag, 
                                    scalar_value, 
                                    global_step, 
                                    time.time())

    def export_scalars_to_json(self, path):
        """Exports to the given path an ASCII file containing all the scalars
        written so far by this instance, with the following format:
        {writer_id : [[timestamp, step, value], ...], ...}
        """
        with open(path, "w") as f:
            json.dump(self.scalar_dict, f)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        """Add histogram to summary."""
        if bins == 'tensorflow':
            bins = self.default_bins
        self._file_writer.add_summary(summary.histogram(tag, values, bins), 
                                    global_step)

    def add_image(self, tag, img_tensor, global_step=None):
        """Add image data to summary."""
        self._file_writer.add_summary(summary.image(tag, img_tensor), 
                                    global_step)

    # X: add text
    def add_text(self, tag, text_string, global_step=None):
        """Add text data to summary."""
        self._file_writer.add_summary(summary.text(tag, text_string), global_step)
        # X: seems like all the text tags are added to a json file
        if tag not in self.text_tags:
            self.text_tags.append(tag)
            if not self.text_dir:
                text_dir =os.path.join(self._file_writer.get_logdir(),
                                        'plugins',
                                        'tensorboard_text')
                os.makedirs(text_dir)
            with open(os.path.join(text_dir, 'tensors.json'), 'w') as fp:
                json.dump(self.text_tags, fp)

    # X: graph is the last part
    def add_graph(self, model):
        self._file_writer.add_graph(graph(model))

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100):
        raise NotImplementedError

    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None):
        raise NotImplementedError

    def add_graph_torch(self, model, input_to_model, verbose=False):
        # prohibit second call?
        # no, let tensorboard handles it and show its warning message.
        """Add graph data to summary.

        Args:
            model (torch.nn.Module): model to draw.
            input_to_model (torch.autograd.Variable): a variable or a tuple of variables to be fed.

        """
        import torch
        from distutils.version import LooseVersion
        # X: interesting, loose version can be used to compare versions
        if LooseVersion(torch.__version__) >= LooseVersion("0.3.1"):
            pass
        else:
            if LooseVersion(torch.__version__) >= LooseVersion("0.3.0"):
                print('You are using PyTorch==0.3.0, use add_graph_onnx()')
                return
            if not hasattr(torch.autograd.Variable, 'grad_fn'):
                print('add_graph() only supports PyTorch v0.2.')
                return
        self._file_writer.add_graph(graph_torch(model, input_to_model, verbose))

    def close(self):
        if not self._file_writer._closed:
            self._file_writer.flush()
            self._file_writer.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
