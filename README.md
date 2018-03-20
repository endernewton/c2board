# c2board

A hacked-up visualization tool for [caffe2](https://caffe2.ai/). Specifically, it dumps the computation graph and the training statistics into a [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) compatible format. Once it starts dumping, you can use tensorboard to visualize the results.

### Prerequisites

- Caffe2.
- Tensorboard. The code is meaned to be standalone, so that you do not need to import **both** [tensorflow](https://www.tensorflow.org/) and caffe2 at the same time -- unexpected behaviors can occur. However, tensorboard needs to be installed somewhere (like in another [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)) for visualization, otherwise the dumped information is not useful by itself.
- Python libraries:

  ```Shell
  # for conda users
  conda install protobuf
  # or for ubuntu: sudo apt-get install protobuf-compiler libprotobuf-dev
  pip install json numpy Pillow six threading
  ```

### Installation

1. Clone the repository
  ```Shell
  git clone https://github.com/endernewton/c2board.git
  ```
2. Make and install it locally
  ```Shell
  cd c2board
  make
  ```

### Usage

- From graph visualization. You can follow demo_graph.py.

### Screenshots

### References

- [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) to make tensors flow without tensorflow.
- [caffe2-native tensorboard](https://github.com/caffe2/caffe2/tree/master/caffe2/contrib/tensorboard) to visualize graphs.