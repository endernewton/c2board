'''Provides an API for generating Event protocol buffers.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re
import six
import time

from caffe2.python import cnn, core
from caffe2.proto import caffe2_pb2

from c2board.src import event_pb2
from c2board.src import summary_pb2
from c2board.src import graph_pb2
from c2board.event_file_writer import EventFileWriter
from c2board.graph import model_to_graph, nets_to_graph, protos_to_graph
from c2board.x2num import make_nps
import c2board.summary as summary


class FileWriter(object):
    '''Write `Summary` protocol buffers to event files.'''
    def __init__(self, 
                logdir,
                max_queue=10,
                flush_secs=120):
        '''Create a `SummaryWriter` and an event file.'''
        self._event_writer = EventFileWriter(logdir, max_queue, flush_secs)
        self._closed = False

    def get_logdir(self):
        '''Return the directory.'''
        return self._event_writer.get_logdir()

    def add_summary(self, summary, global_step=None):
        '''Add a `Summary` protocol buffer to the event file.'''
        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ
        event = event_pb2.Event(summary=summary)
        self._add_event(event, global_step)

    def add_graph(self, graph):
        '''Add a `Graph` protocol buffer to the event file.'''
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self._add_event(event, None)

    def _add_event(self, event, step):
        '''General function to add the event.'''
        event.wall_time = time.time()
        if step is not None:
            event.step = int(step)
        self._event_writer.add_event(event)

    def flush(self):
        '''Flush the event writer.'''
        self._event_writer.flush()

    def close(self):
        '''Close the file writer.'''
        self._event_writer.close()
        self._closed = True


class SummaryWriter(object):
    '''Write `Summary` directly to event files.'''
    def __init__(self, log_dir=None, tag='default',bins=100):
        '''Initialize the summary writer.'''
        if not log_dir:
            # Default: log to runs/
            log_dir = os.path.join('runs', tag)
        self._file_writer = FileWriter(logdir=log_dir)
        self.histogram_dict = {}
        self.histogram_keys = []
        self.histogram_values = []
        self.default_bins = bins
        self.image_dict = {}
        self.rois_dict = {}
        self.text_dir = None
        self.text_tags = []
        self._track_blob_names = {}
        self._reversed_block_names = {}

    def append_histogram(self, name):
        '''Append the name of the blobs to a list for histograms.'''
        self.histogram_dict[name] = name

    def append_image(self, name):
        '''Append the name of the blobs to a list for images.'''
        self.image_dict[name] = name

    def append_image_boxes(self, im_name, box_name):
        self.histogram_dict[box_name] = box_name
        self.image_dict[im_name] = im_name
        self.rois_dict[im_name] = box_name

    def reverse_map(self):
        '''Reverse the map from the graph.'''
        for key, value in six.iteritems(self._track_blob_names):
            if value in self._reversed_block_names:
                self._reversed_block_names[value].append(key)
            else:
                self._reversed_block_names[value] = [key]

    def check_names(self):
        '''Make sure we do not double dump the blobs.'''
        assert len(self.histogram_dict) == len(set(self.histogram_dict)), \
                "ERROR: duplicate name to account histograms"
        assert len(self.image_dict) == len(set(self.image_dict)), \
                "ERROR: duplicate name to account images"

    def replace_names(self, dictionary):
        '''Replace the names according to the graph.'''
        GPU = re.compile('gpu_[0-9]+/')

        for key in dictionary.keys():
            # Remove GPU information, assume it is data parallelism
            # TODO(xinleic): make it applicable to more general cases
            match = GPU.match(key).group()
            key0 = key.replace(match, 'gpu_0/')
            assert key0 in self._reversed_block_names, \
                             "ERROR: {} not found in blob names!".format(key0)
            values = self._reversed_block_names[key0]
            # Hack, just get the common ones
            value = summary.clean_tag(match + os.path.commonprefix(values))
            dictionary[key] = value

    def sort_out_names(self):
        '''Wrapper function to replace names.'''
        if self._track_blob_names:
            if not self._reversed_block_names:
                self.reverse_map()

            self.replace_names(self.histogram_dict)
            for key, value in six.iteritems(self.histogram_dict):
                self.histogram_keys.append(key)
                self.histogram_values.append(value)
            self.replace_names(self.image_dict)

    def _add_scalar(self, tag, scalar_value, global_step):
        '''Add scalar data to summary.'''
        self._file_writer.add_summary(summary.scalar(tag, scalar_value), 
                                    global_step)

    def _add_histogram(self, tag, values, global_step):
        '''Add histogram to summary.'''
        self._file_writer.add_summary(summary.histogram(tag, values, self.default_bins), 
                                    global_step)

    def _add_histograms(self, global_step):
        '''Add multiple histograms to summary.'''
        values = make_nps(self.histogram_keys)
        for name, value in zip(self.histogram_values, values):
            self._file_writer.add_summary(summary.histogram_with_values(name, 
                                                                        value, 
                                                                        self.default_bins),
                                        global_step)

    def _add_image(self, tag, img_tensor, global_step, **kwargs):
        '''Add image data to summary.'''
        res = summary.image(tag, img_tensor, **kwargs)
        if isinstance(res, list):
            for r in res:
                self._file_writer.add_summary(r, global_step)
        else:
            self._file_writer.add_summary(res, global_step)

    def _add_image_boxes(self, tag, img_tensor, box_tensor, global_step, **kwargs):
        '''Add image data to summary.'''
        res = summary.image_boxes(tag, img_tensor, box_tensor, **kwargs)
        if isinstance(res, list):
            for r in res:
                self._file_writer.add_summary(r, global_step)
        else:
            self._file_writer.add_summary(res, global_step)

    def _add_text(self, tag, text_string, global_step):
        '''Add text data to summary.'''
        self._file_writer.add_summary(summary.text(tag, text_string), global_step)
        if tag not in self.text_tags:
            self.text_tags.append(tag)
            if not self.text_dir:
                text_dir =os.path.join(self._file_writer.get_logdir(),
                                        'plugins',
                                        'tensorboard_text')
                os.makedirs(text_dir)
            with open(os.path.join(text_dir, 'tensors.json'), 'w') as fp:
                json.dump(self.text_tags, fp)

    def add_audio(self, 
                tag, 
                snd_tensor, 
                global_step, 
                sample_rate=44100):
        raise NotImplementedError

    def add_pr_curve(self, 
                    tag, 
                    labels, 
                    predictions, 
                    global_step, 
                    num_thresholds=127, 
                    weights=None):
        raise NotImplementedError

    def write_graph(self, model_or_nets_or_protos=None, **kwargs):
        '''Write graph to the summary.'''
        if isinstance(model_or_nets_or_protos, cnn.CNNModelHelper):
            current_graph, track_blob_names = model_to_graph(model_or_nets_or_protos, **kwargs)
        elif isinstance(model_or_nets_or_protos, list):
            if isinstance(model_or_nets_or_protos[0], core.Net):
                current_graph, track_blob_names = nets_to_graph(model_or_nets_or_protos, **kwargs)
            elif isinstance(model_or_nets_or_protos[0], caffe2_pb2.NetDef):
                current_graph, track_blob_names = protos_to_graph(model_or_nets_or_protos, **kwargs)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self._file_writer.add_graph(current_graph)
        self._track_blob_names = track_blob_names
        # Once the graph is built, one can just map the blobs
        self.check_names()
        self.sort_out_names()

    def write_scalars(self, dictionary, global_step):
        '''Write multiple scalars to summary.'''
        for key, value in six.iteritems(dictionary):
            self._add_scalar(key, value, global_step)

    def write_summaries(self, global_step):
        '''Write histogram and image summaries.'''
        # for key, value in six.iteritems(self.histogram_dict):
        #     self._add_histogram(value, key, global_step)
        self._add_histograms(global_step)
        if self.rois_dict:
            for im_name, box_name in six.iteritems(self.rois_dict):
                self._add_image_boxes(box_name, im_name, box_name, global_step)
        else:
            for key, value in six.iteritems(self.image_dict):
                self._add_image(value, key, global_step)

    def close(self):
        '''Close the writers.'''
        if not self._file_writer._closed:
            self._file_writer.flush()
            self._file_writer.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
