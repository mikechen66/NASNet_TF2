#!/usr/bin/env python
# coding: utf-8

# predict_large.py

"""
NASNet-A models for Keras.

NASNet refers to Neural Architecture Search Network, a family of model that were 
designed automatically by learning the model architectures directly on the dataset
of interest.

Here we consider NASNet-A, the highest performance model that was found for the 
CIFAR-10 dataset, and then extended to ImageNet 2012 dataset, obtaining state of 
the art performance on CIFAR-10 and ImageNet 2012. Only the NASNet-A models, and 
their respective weights, which are suited for ImageNet 2012 are provided.

The below table describes the performance on ImageNet 2012:
--------------------------------------------------------------------------------
      Architecture       | Top-1 Acc | Top-5 Acc |  Multiply-Adds |  Params (M)
--------------------------------------------------------------------------------
|   NASNet-A (4 @ 1056)  |   74.0 %  |   91.6 %  |       564 M    |     5.3    |
|   NASNet-A (6 @ 4032)  |   82.7 %  |   96.2 %  |      23.8 B    |    88.9    |
--------------------------------------------------------------------------------
 
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

import tensorflow as tf 
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from nasnet_func import NASNetLarge


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def preprocess_input(x):
    
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output


if __name__ == '__main__':

    model = NASNetLarge(include_top=True, weights='imagenet')
    
    model.summary()

    img_path = '/home/mike/Documents/keras_nasnet/plane.jpg'
    img = image.load_img(img_path, target_size=(331,331))
    output = preprocess_input(img)
    print('Input image shape:', output.shape)

    preds = model.predict(output)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds,1))