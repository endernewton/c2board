"""
To write tf_record into file. Here we use it for tensorboard's event writing.
The code was borrowed from https://github.com/TeamHG-Memex/tensorboard_logger
"""

import re
import struct

from .crc32c import crc32c

# X: this is the most basic record writer?
class RecordWriter(object):
    def __init__(self, path):
        self._path = path
        self._writer = open(path, 'wb')

    def write(self, event_str):
        w = self._writer.write
        header = struct.pack('Q', len(event_str))
        w(header)
        w(struct.pack('I', masked_crc32c(header)))
        w(event_str)
        w(struct.pack('I', masked_crc32c(event_str)))
        self._writer.flush()

    # X: make sure to close it on exit
    def __del__(self):
        if self._writer is not None:
            self._writer.close()

def masked_crc32c(data):
    x = u32(crc32c(data))
    return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)

def u32(x):
    return x & 0xffffffff
