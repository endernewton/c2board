# X: write our own graph interface
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import bytes
import copy
import os
import six

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

from c2board.src.graph_pb2 import GraphDef
from c2board.src.node_def_pb2 import NodeDef
from c2board.src.versions_pb2 import VersionDef
# X: need to add attribute values
from c2board.src.attr_value_pb2 import AttrValue
from c2board.src.tensor_shape_pb2 import TensorShapeProto

# X: let's first go with a version without shapes
# def _try_get_shapes(nets):
#     try:
#         # Note: this will inspect the workspace for better or worse.
#         _ = workspace.InferShapesAndTypes(nets)
#         return shapes
#     except Exception as e:
#         print('WARNING: Failed to compute shapes: %s', e)
#         return {}

# X: it seems not necessary to propagate it..
def _propagate_device_option(net):
    if not net.HasField("device_option"):
        return
    for op in net.op:
        if not op.HasField("device_option"):
            op.device_option.CopyFrom(net.device_option)

# X: get blob names, not sure if needed
def _get_blob_names(ops):
    names = set()
    for op in ops:
        names.update(op.input)
        names.update(op.output)
    return {name: name for name in names}

# X: wow this is very tensorflow
def _make_unique_name(seen, name, min_version=0):
    assert name is not None
    i = min_version
    x = '%s_%d' % (name, i) if i else name
    while x in seen:
        i += 1
        x = '%s_%d' % (name, i)
    seen.add(x)
    return x

def _remap_keys(m, f):
    m2 = {f(key): value for key, value in six.iteritems(m)}
    m.clear()
    m.update(m2)

# X: rename all the ops
def _rename_all(track_blob_names, ops, f):
    seen = set()
    renamed = {}

    def g(name):
        """ Collision-free version of f.
        """
        if name is None:
            return None
        if name in renamed:
            return renamed[name]
        new_name = _make_unique_name(seen, f(name))
        renamed[name] = new_name
        return new_name

    for op in ops:
        inputs = list(op.input)
        outputs = list(op.output)
        # X: remove all the inputs and outputs
        del op.input[:]
        del op.output[:]
        op.input.extend(g(name) for name in inputs)
        op.output.extend(g(name) for name in outputs)

    if track_blob_names:
        _remap_keys(track_blob_names, g)
    # Rename all operator names (if any) independently so that the
    # unique-fixation happens only once in _fill_missing_operator_names().
    seen.clear()
    renamed.clear()
    for op in ops:
        op.name = g(op.name)

def _replace_colons(track_blob_names, ops, repl):
    """
    `:i` has a special meaning in Tensorflow.
    """
    def f(name):
        return name.replace(':', repl)
    _rename_all(track_blob_names, ops, f)

def _replace_underscores(track_blob_names, ops, repl):
    """
    `:i` has a special meaning in Tensorflow.
    """
    def f(name):
        return name.replace('_', repl)
    _rename_all(track_blob_names, ops, f)

def _convert_to_ssa(track_blob_names, ops):
    """
    Convert an operator graph to SSA (i.e. out-of-place).
    I.e. blobs will be renamed so that each blob is produced only once.
    """
    ir = core.IR(ops)
    seen = set()
    versioned = {}
    new_track_blob_names = {}

    def ssa_name(name, versions):
        assert name in versions
        version = versions[name]
        if (name, version) in versioned:
            return versioned[(name, version)]
        # Always setting new_name = `{name}_{version}` would work, but we also try
        # to avoid a trailing `_0`, so we have to be careful not to introduce
        # name collisions, such as (foo_1, 0) = foo_1 = (foo, 1).
        # Note: operator names (if any) will be handled later.
        new_name = _make_unique_name(seen, name, min_version=version)
        versioned[(name, version)] = new_name
        if track_blob_names and name in track_blob_names:
            new_track_blob_names[new_name] = track_blob_names[name]
        return new_name

    for (op, ssa) in zip(ops, ir.ssa):
        assert op is ssa.op
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        op.input.extend(ssa_name(name, ssa.in_versions) for name in inputs)
        op.output.extend(ssa_name(name, ssa.out_versions) for name in outputs)

    if track_blob_names:
        track_blob_names.clear()
        track_blob_names.update(new_track_blob_names)

def _add_gradient_scope(track_blob_names, ops):
    """
    For all operators or blobs with name containing "_grad", add a
    "GRADIENTS/" scope.
    """
    def f(name):
        if '_grad' in name:
            return 'GRADIENTS/{}'.format(name.replace('_grad',''))
        else:
            return name
    _rename_all(track_blob_names, ops, f)

def _add_momentum_scope(track_blob_names, ops):
    """
    For all operators or blobs with name containing "_momentum", add a
    "MOMENTUM/" scope.
    """
    def f(name):
        if '_momentum' in name:
            return 'MOMENTUM/{}'.format(name.replace('_momentum',''))
        else:
            return name
    _rename_all(track_blob_names, ops, f)

def _fill_missing_operator_names(ops):
    ''' Give missing operators a name.

    We expect C2 operators to be generally unnamed. This gives them a scope
    (inferred from their outputs) and a name after their type. Duplicates will
    be postfixed by an index.
    '''
    seen = set()
    for op in ops:
        # Make sure operator names don't collide with blobs.
        seen.update(op.input)
        seen.update(op.output)
    for op in ops:
        if op.name:
            name = op.name
        elif op.output or op.input:
            # X: a hack to get rid of the added stuff
            l = [os.path.dirname(name) for name in op.output or op.input]
            # X: remove the trailing underscores and numbers,
            # which is an artifact of making the names unique
            scope = os.path.commonprefix(l)
            name = os.path.join(scope, op.type)
        else:
            name = op.type
        assert(name)
        op.name = _make_unique_name(seen, name)

def _tf_device(device_option):
    if not device_option.HasField("device_type"):
        return ""
    if device_option.device_type == caffe2_pb2.CPU:
        return "/cpu:*"
    if device_option.device_type == caffe2_pb2.CUDA:
        return "/gpu:{}".format(device_option.cuda_gpu_id)
    raise Exception("Un-handled device", device_option)

def _add_tf_shape(m, ints):
    sh = TensorShapeProto()
    for i in ints:
        dim = TensorShapeProto.Dim()
        dim.size = i
        sh.dim.extend([dim])
    m['_output_shapes'].list.shape.extend([sh])


def _set_tf_attr(m, arg):
    k = arg.name
    if k == 'shape' and arg.ints:
        _add_tf_shape(m, arg.ints)
        return
    if arg.HasField("f"):
        m[k].f = arg.f
        return
    if arg.HasField("i"):
        m[k].i = arg.i
        return
    if arg.HasField("s"):
        m[k].s = (
            arg.s if isinstance(arg.s, bytes) else str(arg.s).encode('utf-8')
        )
        return
    if arg.floats:
        m[k].list.f.extend(arg.floats)
        return
    if arg.ints:
        m[k].list.i.extend(arg.ints)
        return
    if arg.strings:
        m[k].list.s.extend(
            s if isinstance(s, bytes) else str(s).encode('utf-8')
            for s in arg.strings
        )
        return
    # The value is an empty list.
    m[k].list.s.extend([])

def _operator_to_node(op):
    assert op.name, op
    n = NodeDef()
    n.name = op.name
    n.input.extend(op.input)
    n.op = op.type
    n.device = _tf_device(op.device_option)
    for arg in op.arg:
        _set_tf_attr(n.attr, arg)
    return n

def _blob_to_node(producing_ops, name):
    assert name
    n = NodeDef()
    n.name = name
    inputs = producing_ops.get(name, [])
    if inputs:
        n.op = 'Blob'
    else:
        n.op = 'Placeholder'
    n.input.extend('%s:%d' % (op.name, i) for op, i in inputs)
    if inputs:
        device = inputs[0][0].device_option
        if (all(input[0].device_option == device for input in inputs)):
            n.device = _tf_device(device)
    return n

# X: remove the debug information, they are copious
def _clear_debug_info(ops):
    for op in ops:
        if op.HasField('debug_info'):
            op.ClearField('debug_info')

def _operators_to_graph_def(ops,
                            replace_colons=None,
                            replace_underscores=None,
                            with_ssa=True,
                            with_gradient_scope=True,
                            track_blob_names=None,
                            clear_debug_info=True):
    if clear_debug_info:
        _clear_debug_info(ops)
    if track_blob_names is not None:
        track_blob_names.clear()
        track_blob_names.update(_get_blob_names(ops))
    if replace_colons:
        _replace_colons(track_blob_names, ops, replace_colons)
    if replace_underscores:
        _replace_underscores(track_blob_names, ops, replace_underscores)
    if with_gradient_scope:
        _add_momentum_scope(track_blob_names, ops)
        _add_gradient_scope(track_blob_names, ops)
    # X: this is necessary since caffe can have in-place operator
    if with_ssa:
        _convert_to_ssa(track_blob_names, ops)
    _fill_missing_operator_names(ops)

    # X: apparently the external inputs are missing
    # X: why producer is 22?
    current_graph = GraphDef(versions=VersionDef(producer=22))
    # X: The next stages are just adding nodes
    producing_ops = {}
    blobs = set()
    for op in ops:
        current_graph.node.extend([_operator_to_node(op)])
        for input_blob in op.input:
            blobs.add(input_blob)
        for i, output_blob in enumerate(op.output):
            blobs.add(output_blob)
            # X: to record the op that produces the blob
            producing_ops.setdefault(output_blob, []).append((op, i))
    for blob in blobs:
        current_graph.node.extend([_blob_to_node(producing_ops, blob)])
    return current_graph

def model_to_graph(model):
    # X: for some reason it needs to get the initialization operations as well 
    nets = [model.param_init_net, model.net]
    return nets_to_graph(nets)

def nets_to_graph(nets):
    # X: get the network proto
    nets = [copy.deepcopy(net.Proto()) for net in nets]
    return protos_to_graph(nets)

def protos_to_graph(nets):
    for net in nets:
        _propagate_device_option(net)
    ops = [op for net in nets for op in net.op]
    current_graph = _operators_to_graph_def(ops)
    return current_graph