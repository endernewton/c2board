# c2board

A hacked-up visualization tool for [caffe2](https://caffe2.ai/). Specifically, it dumps the computation graph and the training statistics into a [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) compatible format. Once it starts dumping, you can use tensorboard to visualize the results.

The code is meaned to be standalone, so that you do not need to import **both** [tensorflow](https://www.tensorflow.org/) and caffe2 at the same time -- unexpected behavior can occur. However, tensorboard needs to be **installed somewhere** (like in another [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)) for visualization.

### Installation



### References

- [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) to make tensors flow without tensorflow.
- [caffe2-native tensorboard](https://github.com/caffe2/caffe2/tree/master/caffe2/contrib/tensorboard) to visualize graphs.