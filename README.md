# nasnet_tf2

NASNet-A models for Keras.

NASNet refers to Neural Architecture Search Network, a family of model that were 
designed automatically by learning the model architectures directly on the dataset
of interest.

Here we consider NASNet-A, the highest performance model that was found for the 
CIFAR-10 dataset, and then extended to ImageNet 2012 dataset, obtaining state of 
the art performance on CIFAR-10 and ImageNet 2012. Only the NASNet-A models, and 
their respective weights, which are suited for ImageNet 2012 are provided.
 
Make the the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 
2.4.3, CUDA Toolkit 11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new 
ines of code to replace the deprecated code. 

Weights obtained from the official TensorFlow repository found at
https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet

# References
 - [Learning Transferable Architectures for Scalable Image Recognition]
    (https://arxiv.org/abs/1707.07012) (CVPR 2018)

This model is based on the following implementations:
 - [TF Slim Implementation]
   (https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py)
 - [TensorNets implementation]
   (https://github.com/taehoonlee/tensornets/blob/master/tensornets/nasnets.py)
"""
