"""Provides an API for generating Event protocol buffers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re
import six
import time

from c2board.src import event_pb2
from c2board.src import summary_pb2
from c2board.src import graph_pb2
from c2board.x2num import make_np

from c2board.event_file_writer import EventFileWriter
from c2board.graph import model_to_graph, nets_to_graph, protos_to_graph
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
        self.histogram_dict = {}
        self.image_dict = {}
        self.text_dir = None
        self._track_blob_names = {}
        self._reversed_block_names = {}

    def append_scalar(self, name):
        self.scalar_dict[name] = []

    # X: this is done during the construction of the graph, so just append
    def append_histogram(self, name):
        # X: later it can be mapped to many names
        self.histogram_dict[name] = name

    def append_image(self, name):
        self.image_dict[name] = name

    def reverse_map(self):
        for key, value in six.iteritems(self._track_blob_names):
            if value in self._reversed_block_names:
                self._reversed_block_names[value].append(key)
            else:
                self._reversed_block_names[value] = [key]

    # X: first need to check we do not double dump the blobs
    def check_names(self):
        assert len(self.histogram_dict) == len(set(self.histogram_dict)), \
                "ERROR: duplicate name to account histograms"
        assert len(self.image_dict) == len(set(self.image_dict)), \
                "ERROR: duplicate name to account images"

    def replace_names(self, dictionary):
        GPU = re.compile('gpu_[0-9]+/')

        for key in dictionary.keys():
            # X: remove GPU information, assume it is data parallelism
            # TODO: make it applicable to everything
            match = GPU.match(key).group()
            key0 = key.replace(match, 'gpu_0/')
            assert key0 in self._reversed_block_names, \
                             "ERROR: {} not found in blob names!".format(key0)
            values = self._reversed_block_names[key0]
            # X: hack, just get the common ones
            value = summary.clean_tag(match + os.path.commonprefix(values))
            dictionary[key] = value

    def sort_out_names(self):
        if self._track_blob_names:
            if not self._reversed_block_names:
                self.reverse_map()

            self.replace_names(self.histogram_dict)
            self.replace_names(self.image_dict)

    def add_scalar(self, tag, scalar_value, global_step):
        """Add scalar data to summary.
        """
        self._file_writer.add_summary(summary.scalar(tag, scalar_value), 
                                    global_step)

    def add_histogram(self, tag, values, global_step, bins='tensorflow'):
        """Add histogram to summary."""
        if bins == 'tensorflow':
            bins = self.default_bins
        self._file_writer.add_summary(summary.histogram(tag, values, bins), 
                                    global_step)

    def add_image(self, tag, img_tensor, global_step):
        """Add image data to summary."""
        self._file_writer.add_summary(summary.image(tag, img_tensor), 
                                    global_step)

    # X: add text
    def add_text(self, tag, text_string, global_step):
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
    def add_graph(self, model=None, nets=None, protos=None, **kwargs):
        if not model and not nets and not protos:
            raise ValueError("input must be either a model or a list of nets")
        if model:
            current_graph, track_blob_names = model_to_graph(model, **kwargs)
        elif nets:
            current_graph, track_blob_names = nets_to_graph(nets, **kwargs)
        else:
            current_graph, track_blob_names = protos_to_graph(protos, **kwargs)
        self._file_writer.add_graph(current_graph)
        self._track_blob_names = track_blob_names
        # X: once the graph is built, one can just map the blobs
        self.check_names()
        self.sort_out_names()

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100):
        raise NotImplementedError

    def add_pr_curve(self, tag, labels, predictions, 
                    global_step=None, 
                    num_thresholds=127, 
                    weights=None):
        raise NotImplementedError

    # X: the function to call to dump the values
    def write_summaries(self, global_step):
        for key, value in six.iteritems(self.histogram_dict):
            self.add_histogram(value, key, global_step)
        for key, value in six.iteritems(self.image_dict):
            self.add_image(value, key, global_step)

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
