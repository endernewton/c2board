# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: c2board/src/plugin_pr_curve.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='c2board/src/plugin_pr_curve.proto',
  package='tensorboard',
  syntax='proto3',
  serialized_pb=_b('\n!c2board/src/plugin_pr_curve.proto\x12\x0btensorboard\"<\n\x11PrCurvePluginData\x12\x0f\n\x07version\x18\x01 \x01(\x05\x12\x16\n\x0enum_thresholds\x18\x02 \x01(\rb\x06proto3')
)




_PRCURVEPLUGINDATA = _descriptor.Descriptor(
  name='PrCurvePluginData',
  full_name='tensorboard.PrCurvePluginData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='version', full_name='tensorboard.PrCurvePluginData.version', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_thresholds', full_name='tensorboard.PrCurvePluginData.num_thresholds', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=110,
)

DESCRIPTOR.message_types_by_name['PrCurvePluginData'] = _PRCURVEPLUGINDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PrCurvePluginData = _reflection.GeneratedProtocolMessageType('PrCurvePluginData', (_message.Message,), dict(
  DESCRIPTOR = _PRCURVEPLUGINDATA,
  __module__ = 'c2board.src.plugin_pr_curve_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.PrCurvePluginData)
  ))
_sym_db.RegisterMessage(PrCurvePluginData)


# @@protoc_insertion_point(module_scope)