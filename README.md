# NASNet_TF2

NASNet-A models for Keras.

NASNet refers to Neural Architecture Search Network, a family of model that were 
designed automatically by learning the model architectures directly on the dataset
of interest.

It includes the two motifs of Normal Cell and Reduction Cell where the conv blocks 
are constructed by Controller Model Architecture. It combines both the RNN and the 
CNN for a better classification and prediction on the images given by users. 
 
Make the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 
2.4.3, CUDA Toolkit 11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new 
lines of code to replace the deprecated code. 

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
