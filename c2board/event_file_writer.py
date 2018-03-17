"""Writes events to disk in a logdir."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import socket
import struct
import threading
import time

from c2board.crc32c import crc32c
from c2board.src import event_pb2


# X: first define necessary functions to write the events
def masked_crc32c(data):
    x = u32(crc32c(data))
    return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)

def u32(x):
    return x & 0xffffffff


# X: this script will map the event to strings
class EventsWriter(object):
    '''Writes `Event` protocol buffers to an event file.'''

    def __init__(self, file_prefix='events'):
        '''
        Events files have a name of the form
        '/some/file/path/events.out.tfevents.[timestamp].[hostname]'
        '''
        self._path = file_prefix + ".out.tfevents." + \
                            str(time.time())[:10] + \
                            "." + socket.gethostname()

        self._num_outstanding_events = 0

        # X: open up the writer
        self._writer = open(self._path, 'wb')

        # Initialize an event instance.
        self._event = event_pb2.Event()

        # X: get the time
        self._event.wall_time = time.time()

        # write event 
        self.write_event(self._event)

    def write_event(self, event):
        '''Append "event" to the file.'''

        # Check if event is of type event_pb2.Event proto.
        if not isinstance(event, event_pb2.Event):
            raise TypeError("Expected an event_pb2.Event proto, "
                            " but got %s" % type(event))
        return self._write_serialized_event(event.SerializeToString())

    def _write_serialized_event(self, event_str):
        self._num_outstanding_events += 1

        # X: do the actual writing, simplified version
        header = struct.pack('Q', len(event_str))
        self._writer.write(header)
        self._writer.write(struct.pack('I', masked_crc32c(header)))
        self._writer.write(event_str)
        self._writer.write(struct.pack('I', masked_crc32c(event_str)))

    def flush(self):
        '''Flushes the events to disk.'''
        self._writer.flush()
        self._num_outstanding_events = 0

    def close(self):
        '''Call self.flush().'''
        if self._writer is not None:
            self.flush()
            self._writer.close()

    # X: make sure to close it on exit
    def __del__(self):
        self.close()


class _EventLoggerThread(threading.Thread):
    """Thread that logs events."""

    def __init__(self, queue, ev_writer, flush_secs=120):
        """Creates an _EventLoggerThread."""
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = queue
        self._ev_writer = ev_writer
        self._flush_secs = flush_secs
        # The first event will be flushed immediately.
        self._next_event_flush_time = 0

    def run(self):
        while True:
            event = self._queue.get()
            try:
                self._ev_writer.write_event(event)
                # Flush the event writer every so often.
                now = time.time()
                # X: only flush it from time to time
                if now > self._next_event_flush_time:
                    self._ev_writer.flush()
                    # Do it again in two minutes.
                    self._next_event_flush_time = now + self._flush_secs
            finally:
                self._queue.task_done()


class EventFileWriter(object):
    """Writes `Event` protocol buffers to an event file.
    The `EventFileWriter` class creates an event file in the specified directory,
    and asynchronously writes Event protocol buffers to the file. The Event file
    is encoded using the tfrecord format, which is similar to RecordIO.
    """

    def __init__(self, logdir, max_queue=10, flush_secs=120):
        """Creates a `EventFileWriter` and an event file to write to."""
        self._logdir = logdir
        # X: if the log file does not exist, create it
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self._event_queue = six.moves.queue.Queue(max_queue)
        self._ev_writer = EventsWriter(os.path.join(self._logdir, "events"))
        self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                          flush_secs)

        # X: assume the thread will automatically call the function run()
        self._worker.start()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def add_event(self, event):
        """Adds an event to the event file.
        Args:
          event: An `Event` protocol buffer.
        """
        self._event_queue.put(event)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to
        disk.
        """
        self._event_queue.join()
        self._ev_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self.flush()
        self._ev_writer.close()
