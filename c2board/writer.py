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


class SummaryWriter(object):
    """Writes `Summary` directly to event files."""
    def __init__(self, log_dir=None, tag='default'):
        if not log_dir:
            # X: just create a name for the log files
            log_dir = os.path.join('runs', tag)
        self._file_writer = FileWriter(logdir=log_dir)
        # X: next is to create bins
        v = 1E-12
        buckets = []
        neg_buckets = []
        while v < 1E20:
            buckets.append(v)
            neg_buckets.append(-v)
            v *= 1.1
        self.default_bins = neg_buckets[::-1] + [0] + buckets
        self.text_tags = []
        self.all_writers = {self._file_writer.get_logdir(): self._file_writer}
        # {writer_id : [[timestamp, step, value],...],...}
        self.scalar_dict = {} 

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

        Args:
            tag (string): Data identifier
            scalar_value (float): Value to save
            global_step (int): Global step value to record
        """
        self._file_writer.add_summary(summary.scalar(tag, scalar_value), global_step)
        self.__append_to_scalar_dict(tag, scalar_value, global_step, time.time())

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        """Adds many scalar data to summary.

        Args:
            tag (string): Data identifier
            main_tag (string): The parent name for the tags
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record

        Examples::

            writer.add_scalars('run_14h',{'xsinx':i*np.sin(i/r),
                                          'xcosx':i*np.cos(i/r),
                                          'arctanx': numsteps*np.arctan(i/r)}, i)
            # This function adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.
        """
        timestamp = time.time()
        fw_logdir = self._file_writer.get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = fw_logdir + "/" + main_tag + "/" + tag
            if fw_tag in self.all_writers.keys():
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(logdir=fw_tag)
                self.all_writers[fw_tag] = fw
            fw.add_summary(summary.scalar(main_tag, scalar_value), global_step)
            self.__append_to_scalar_dict(fw_tag, scalar_value, global_step, timestamp)

    def export_scalars_to_json(self, path):
        """Exports to the given path an ASCII file containing all the scalars written
        so far by this instance, with the following format:
        {writer_id : [[timestamp, step, value], ...], ...}
        """
        with open(path, "w") as f:
            json.dump(self.scalar_dict, f)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        """Add histogram to summary.

        Args:
            tag (string): Data identifier
            values (numpy.array): Values to build histogram
            global_step (int): Global step value to record
            bins (string): one of {'tensorflow','auto', 'fd', ...}, this determines how the bins are made. You can find
              other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        """
        if bins == 'tensorflow':
            bins = self.default_bins
        self._file_writer.add_summary(summary.histogram(tag, values, bins), global_step)

    def add_image(self, tag, img_tensor, global_step=None):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor): Image data
            global_step (int): Global step value to record
        Shape:
            img_tensor: :math:`(3, H, W)`. Use ``torchvision.utils.make_grid()`` to prepare it is a good idea.
        """
        self._file_writer.add_summary(summary.image(tag, img_tensor), global_step)

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100):
        """Add audio data to summary.

        Args:
            tag (string): Data identifier
            snd_tensor (torch.Tensor): Sound data
            global_step (int): Global step value to record
            sample_rate (int): sample rate in Hz

        Shape:
            snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
        """
        self._file_writer.add_summary(summary.audio(tag, snd_tensor, sample_rate=sample_rate), global_step)

    def add_text(self, tag, text_string, global_step=None):
        """Add text data to summary.

        Args:
            tag (string): Data identifier
            text_string (string): String to save
            global_step (int): Global step value to record

        Examples::

            writer.add_text('lstm', 'This is an lstm', 0)
            writer.add_text('rnn', 'This is an rnn', 10)
        """
        self._file_writer.add_summary(summary.text(tag, text_string), global_step)
        if tag not in self.text_tags:
            self.text_tags.append(tag)
            extension_dir = self._file_writer.get_logdir() + '/plugins/tensorboard_text/'
            if not os.path.exists(extension_dir):
                os.makedirs(extension_dir)
            with open(extension_dir + 'tensors.json', 'w') as fp:
                json.dump(self.text_tags, fp)

    def add_graph(self, model, input_to_model, verbose=False):
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
        self._file_writer.add_graph(graph(model, input_to_model, verbose))

    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None):
        """Adds precision recall curve.

        Args:
            tag (string): Data identifier
            labels (torch.Tensor): Ground truth data. Binary label for each element.
            predictions (torch.Tensor): The probability that an element be classified as true. Value should in [0, 1]
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.

        """
        from .x2num import make_np
        labels = make_np(labels)
        predictions = make_np(predictions)
        self._file_writer.add_summary(summary.pr_curve(tag, labels, predictions, num_thresholds, weights), global_step)

    def close(self):
        if self._file_writer is None:
            return  # ignore double close
        self._file_writer.flush()
        self._file_writer.close()
        for path, writer in self.all_writers.items():
            writer.flush()
            writer.close()
        self._file_writer = self.all_writers = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
