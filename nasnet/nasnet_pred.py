#!/usr/bin/env python
# coding: utf-8

# nasnet_pred.py

"""
NASNet-A models for Keras.

NASNet refers to Neural Architecture Search Network, a family of models that were 
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


import os
import math
import warnings
import numpy as np
import tensorflow as tf 

from keras.preprocessing import image
from keras.layers import add, multiply, concatenate
from keras.layers import Conv2D, Input, Dense, Dropout, Reshape, Cropping2D, Activation, \
    DepthwiseConv2D, SeparableConv2D, BatchNormalization, ZeroPadding2D, MaxPooling2D,  \
    AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.models import Model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs

from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from imagenet_utils import _obtain_input_shape


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


BASE_WEIGHTS_PATH = ('https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/')
NASNET_MOBILE_WEIGHT_PATH = BASE_WEIGHTS_PATH + 'NASNet-mobile.h5'
NASNET_MOBILE_WEIGHT_PATH_NO_TOP = BASE_WEIGHTS_PATH + 'NASNet-mobile-no-top.h5'
NASNET_LARGE_WEIGHT_PATH = BASE_WEIGHTS_PATH + 'NASNet-large.h5'
NASNET_LARGE_WEIGHT_PATH_NO_TOP = BASE_WEIGHTS_PATH + 'NASNet-large-no-top.h5'


def correct_pad(K, inputs, kernel_size):
    # Return a tuple for zero-padding for 2D convolution with downsampling.
    """
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 1 if K.image_data_format() == 'channels_last' else 2
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0]%2, 1 - input_size[1]%2)

    correct = (kernel_size[0]//2, kernel_size[1]//2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def _separable_conv_block(ip, filters, kernel_size=(3,3), strides=(1,1), block_id=None):
    # Adds the two blocks of [relu-separable conv-batchnorm].
    """
    # Arguments
        ip: Input tensor
        filters: Number of output filters per layer
        kernel_size: Kernel size of separable convolutions
        strides: Strided convolution for downsampling
        block_id: String block_id
    # Returns
        A Keras tensor
    """
    channel_dim = 3 if K.image_data_format() == 'channels_last' else 1

    with K.name_scope('separable_conv_block_%s' % block_id):
        x = Activation('relu')(ip)
        if strides == (2, 2):
            x = ZeroPadding2D(padding=correct_pad(K,x,kernel_size), 
                              name='separable_conv_1_pad_%s' % block_id)(x)
            conv_pad = 'valid'
        else:
            conv_pad = 'same'
        x = SeparableConv2D(filters, kernel_size, strides=strides, name='separable_conv_1_%s' % block_id, 
                            padding=conv_pad, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, 
                               name='separable_conv_1_bn_%s' % (block_id))(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, kernel_size, name='separable_conv_2_%s' % block_id,  padding='same', 
                            use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, 
                               name='separable_conv_2_bn_%s' % (block_id))(x)

    return x


def _adjust_block(p, ip, filters, block_id=None):
    # Adjusts the input `previous path` to match the shape of the `input`, 
    # Used in the changed the output number of filters.
    """
    # Arguments
        p: Input tensor which needs to be modified
        ip: Input tensor whose shape needs to be matched
        filters: Number of output filters to be matched
        block_id: String block_id
    # Returns
        Adjusted Keras tensor
    """
    # -channel_dim = -1 if K.image_data_format() == 'channels_last' else 1
    channel_dim = 3 if K.image_data_format() == 'channels_last' else 1
    # -img_dim = 2 if K.image_data_format() == 'channels_first' else -2
    img_dim = 1 if K.image_data_format() == 'channels_last' else 2
    ip_shape = K.int_shape(ip)

    if p is not None:
        p_shape = K.int_shape(p)

    with K.name_scope('adjust_block'):
        if p is None:
            p = ip

        elif p_shape[img_dim] != ip_shape[img_dim]:

            with K.name_scope('adjust_reduction_block_%s' % block_id):
                p = Activation('relu', name='adjust_relu_1_%s' % block_id)(p)
                p1 = AveragePooling2D((1,1), strides=(2,2), padding='valid', name='adjust_avg_pool_1_%s' % block_id)(p)
                p1 = Conv2D(filters//2, (1,1), padding='same', use_bias=False, 
                            name='adjust_conv_1_%s' % block_id, kernel_initializer='he_normal')(p1)
                p2 = ZeroPadding2D(padding=((0,1), (0,1)))(p)
                p2 = Cropping2D(cropping=((1,0), (1,0)))(p2)
                p2 = AveragePooling2D((1,1), strides=(2,2), padding='valid', name='adjust_avg_pool_2_%s' % block_id)(p2)
                p2 = Conv2D(filters//2, (1,1), padding='same', use_bias=False, 
                            name='adjust_conv_2_%s' % block_id, kernel_initializer='he_normal')(p2)
                p = concatenate([p1,p2], axis=channel_dim)
                p = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='adjust_bn_%s' % block_id)(p)

        elif p_shape[channel_dim] != filters:

            with K.name_scope('adjust_projection_block_%s' % block_id):
                p = Activation('relu')(p)
                p = Conv2D(filters, (1,1), strides=(1,1), padding='same', name='adjust_conv_projection_%s' % block_id, 
                           use_bias=False, kernel_initializer='he_normal')(p)
                p = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='adjust_bn_%s' % block_id)(p)

    return p


def _normal_a_cell(ip, p, filters, block_id=None):
    # Adds a Normal Cell for NASNet-A (Fig. 4 in the paper).
    """
    # Arguments
        ip: Input tensor `x`
        p: Input tensor `p`
        filters: Number of output filters
        block_id: String block_id
    # Returns
        A Keras tensor
    """
    channel_dim = 3 if K.image_data_format() == 'channels_last' else 1

    with K.name_scope('normal_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, block_id)
        h = Activation('relu')(ip)
        h = Conv2D(filters, (1,1), strides=(1,1), padding='same', name='normal_conv_1_%s' % block_id, 
                   use_bias=False, kernel_initializer='he_normal')(h)
        h = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3,name='normal_bn_1_%s' % block_id)(h)
        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters,kernel_size=(5, 5), block_id='normal_left1_%s' % block_id)
            x1_2 = _separable_conv_block(p, filters, block_id='normal_right1_%s' % block_id)
            x1 = add([x1_1, x1_2], name='normal_add_1_%s' % block_id)

        with K.name_scope('block_2'):
            x2_1 = _separable_conv_block(p, filters, (5,5), block_id='normal_left2_%s' % block_id)
            x2_2 = _separable_conv_block(p, filters, (3,3), block_id='normal_right2_%s' % block_id)
            x2 = add([x2_1, x2_2], name='normal_add_2_%s' % block_id)

        with K.name_scope('block_3'):
            x3 = AveragePooling2D((3,3), strides=(1,1), padding='same', name='normal_left3_%s' % (block_id))(h)
            x3 = add([x3, p], name='normal_add_3_%s' % block_id)

        with K.name_scope('block_4'):
            x4_1 = AveragePooling2D((3,3), strides=(1,1), padding='same', name='normal_left4_%s' % (block_id))(p)
            x4_2 = AveragePooling2D((3,3), strides=(1,1), padding='same', name='normal_right4_%s' % (block_id))(p)
            x4 = add([x4_1, x4_2], name='normal_add_4_%s' % block_id)

        with K.name_scope('block_5'):
            x5 = _separable_conv_block(h, filters, block_id='normal_left5_%s' % block_id)
            x5 = add([x5, h], name='normal_add_5_%s' % block_id)

        x = concatenate([p,x1,x2,x3,x4,x5], axis=channel_dim, name='normal_concat_%s' % block_id)

    return x, ip


def _reduction_a_cell(ip, p, filters, block_id=None):
    # Adds a Reduction cell for NASNet-A (Fig. 4 in the paper).
    """
    # Arguments
        ip: Input tensor `x`
        p: Input tensor `p`
        filters: Number of output filters
        block_id: String block_id
    # Returns
        A Keras tensor
    """
    channel_dim = 3 if K.image_data_format() == 'channels_last' else 1

    with K.name_scope('reduction_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, block_id)
        h = Activation('relu')(ip)
        h = Conv2D(filters, (1,1), strides=(1,1), padding='same', name='reduction_conv_1_%s' % block_id, 
                   use_bias=False, kernel_initializer='he_normal')(h)
        h = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='reduction_bn_1_%s' % block_id)(h)
        h3 = ZeroPadding2D(padding=correct_pad(K,h,3), name='reduction_pad_1_%s' % block_id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, (5,5), strides=(2,2), block_id='reduction_left1_%s' % block_id)
            x1_2 = _separable_conv_block(p, filters, (7,7), strides=(2,2), block_id='reduction_right1_%s' % block_id)
            x1 = add([x1_1, x1_2], name='reduction_add_1_%s' % block_id)

        with K.name_scope('block_2'):
            x2_1 = MaxPooling2D((3,3), strides=(2,2), padding='valid', name='reduction_left2_%s' % block_id)(h3)
            x2_2 = _separable_conv_block(p, filters, (7,7), strides=(2,2), block_id='reduction_right2_%s' % block_id)
            x2 = add([x2_1, x2_2], name='reduction_add_2_%s' % block_id)

        with K.name_scope('block_3'):
            x3_1 = AveragePooling2D((3,3), strides=(2,2), padding='valid', name='reduction_left3_%s' % block_id)(h3)
            x3_2 = _separable_conv_block(p, filters, (5,5), strides=(2,2), block_id='reduction_right3_%s' % block_id)
            x3 = add([x3_1, x3_2], name='reduction_add3_%s' % block_id)

        with K.name_scope('block_4'):
            x4 = AveragePooling2D((3,3), strides=(1,1), padding='same', name='reduction_left4_%s' % block_id)(x1)
            x4 = add([x2, x4])

        with K.name_scope('block_5'):
            x5_1 = _separable_conv_block(x1, filters, (3,3), block_id='reduction_left4_%s' % block_id)
            x5_2 = MaxPooling2D((3,3),strides=(2,2), padding='valid', name='reduction_right5_%s' % block_id)(h3)
            x5 = add([x5_1, x5_2], name='reduction_add4_%s' % block_id)

        x = concatenate([x2,x3,x4,x5], axis=channel_dim, name='reduction_concat_%s' % block_id)

        return x, ip


def NASNet(input_shape=None, penultimate_filters=4032, num_blocks=6, stem_block_filters=96,
           skip_reduction=True, filter_multiplier=2, include_top=True, weights=None,
           input_tensor=None, pooling=None, num_classes=1000, default_size=None, **kwargs):
    # Instantiates a NASNet model.
    """
    # Arguments
        input_shape: tuple, defaulted as `(331,331,3)` for NASNetLarge and
            `(224,224,3)` for NASNetMobile.
        penultimate_filters: filters # in the penultimate layer, notated `NASNet (N @ P)`
            - N is the number of blocks
            - P is the number of penultimate filters
        num_blocks: repeated blocks # of the NASNet model.
        stem_block_filters: filters # in the initial stem block
        skip_reduction: Whether to skip the reduction step at the tail.
        filter_multiplier: Controls the width of the network.
            - If `filter_multiplier` < 1.0,proportionally decreases filters # in each layer.
            - If `filter_multiplier` > 1.0, proportionally increases filters # in each layer
            - If `filter_multiplier` = 1, default filters # from the paper used at each layer.
        include_top: whether to include the FC layer at the top of the network.
        weights: `None` (random initialization) or 'imagenet'. 
        input_tensor: optional Keras tensor (output of `layers.Input()`)
            to use as image input for the model.
        pooling: Optional mode for feature extraction when `include_top` is `False`.
            - `None`: the output of model is the 4D tensor of the last conv layer 
            - `avg` means global average pooling and the output as a 2D tensor.
            - `max` means global max pooling will be applied.
        num_classes: specified if `include_top` is True
        default_size: Specifies the default image size of the model
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights` or invalid input shape.
        RuntimeError: run the model with a backend without support separable conv.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and num_classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    if (isinstance(input_shape, tuple) and None in input_shape and 
            weights == 'imagenet'):
        raise ValueError('When specifying the input shape of a NASNet'
                         ' and loading `ImageNet` weights, '
                         'the input_shape argument must be static '
                         '(no None entries). Got: `input_shape=' +
                         str(input_shape) + '`.')

    if default_size is None:
        default_size = 331

    # Determine the proper input shape and default size.
    input_shape = _obtain_input_shape(input_shape, default_size=default_size, min_size=32,
                                      data_format=K.image_data_format(), 
                                      require_flatten=True, weights=weights)

    if K.image_data_format() != 'channels_last':
        warnings.warn('The NASNet family of models is only available '
                      'for the input data format "channels_last" ')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if penultimate_filters % (24*(filter_multiplier**2)) != 0:
        raise ValueError(
            'For NASNet-A models, the `penultimate_filters` must be a multiple '
            'of 24 * (`filter_multiplier` ** 2). Current value: %d' %
            penultimate_filters)

    channel_dim = 3 if K.image_data_format() == 'channels_last' else 1
    filters = penultimate_filters // 24

    x = Conv2D(stem_block_filters, (3,3), strides=(2,2), padding='valid', use_bias=False,
               name='stem_conv1', kernel_initializer='he_normal')(img_input)
    x = BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='stem_bn1')(x)

    p = None
    x, p = _reduction_a_cell(x, p, filters // (filter_multiplier**2), block_id='stem_1')
    x, p = _reduction_a_cell(x, p, filters // filter_multiplier, block_id='stem_2')

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters, block_id='%d' % (i))

    x, p0 = _reduction_a_cell(x, p, filters*filter_multiplier, block_id='reduce_%d' % (num_blocks))
    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters*filter_multiplier, block_id='%d' % (num_blocks+i+1))

    x, p0 = _reduction_a_cell(x, p, filters*filter_multiplier**2, block_id='reduce_%d' % (2*num_blocks))
    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters*filter_multiplier**2, block_id='%d' % (2*num_blocks+i+1))

    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(num_classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure the model considers any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='NASNet')

    # Load the 4 weights.
    if weights == 'imagenet':
        if default_size == 224:  # mobile version
            if include_top:
                weights_path = get_file(
                    'nasnet_mobile.h5',
                    NASNET_MOBILE_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='020fb642bf7360b370c678b08e0adf61')
            else:
                weights_path = get_file(
                    'nasnet_mobile_no_top.h5',
                    NASNET_MOBILE_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='1ed92395b5b598bdda52abe5c0dbfd63')
            model.load_weights(weights_path)
        elif default_size == 331:  # large version
            if include_top:
                weights_path = get_file(
                    'nasnet_large.h5',
                    NASNET_LARGE_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='11577c9a518f0070763c2b964a382f17')
            else:
                weights_path = get_file(
                    'nasnet_large_no_top.h5',
                    NASNET_LARGE_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='d81d89dc07e6e56530c4e77faddd61b5')
            model.load_weights(weights_path)
        else:
            raise ValueError(
                'ImageNet weights can only be loaded with NASNetLarge'
                ' or NASNetMobile')
    elif weights is not None:
        model.load_weights(weights)

    if old_data_format:
        K.set_image_data_format(old_data_format)

    return model


def NASNetLarge(input_shape=None, include_top=True, weights='imagenet',
                input_tensor=None, pooling=None, num_classes=1000, **kwargs):
    # Instantiate the large NASNet model in the ImageNet mode.
    """
    # Arguments
        input_shape: tuple, defaulted as `(331,331,3)` for NASNetLarge and
            `(224,224,3)` for NASNetMobile.
        include_top: whether to include the FC layer at the top of the network.
        weights: `None` (random initialization)or 'imagenet'. 
        input_tensor: optional Keras tensor (output of `layers.Input()`)
            to use as image input for the model.
       pooling: Optional mode for feature extraction when `include_top` is `False`.
            - `None`: the output of model is the 4D tensor of the last conv layer 
            - `avg` means global average pooling and the output as a 2D tensor.
            - `max` means global max pooling will be applied.
        num_classes: specified if `include_top` is True
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    return NASNet(input_shape, penultimate_filters=4032, num_blocks=6, stem_block_filters=96,
                  skip_reduction=True, filter_multiplier=2, include_top=include_top, 
                  weights=weights, input_tensor=input_tensor, pooling=pooling, 
                  num_classes=num_classes, default_size=331, **kwargs)


def NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None,
                 pooling=None, num_classes=1000, **kwargs):
    # Instantiate the Mobile NASNet model in the ImageNet mode
    """
    # Arguments
       input_shape: tuple, defaulted as `(224,224,3)` for NASNetMobile.
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the FC layer at the top of the network.
        weights: `None` (random initialization)or 'imagenet'. 
        input_tensor: optional Keras tensor (output of `layers.Input()`)
            to use as image input for the model.
        pooling: Optional mode for feature extraction when `include_top` is `False`.
            - `None`: the output of model is the 4D tensor of the last conv layer 
            - `avg` means global average pooling and the output as a 2D tensor.
            - `max` means global max pooling will be applied.
        num_classes: specified if `include_top` is True
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights` or invalid input shape.
        RuntimeError: run the model with a backend without support separable conv.
    """
    return NASNet(input_shape, penultimate_filters=1056, num_blocks=4, stem_block_filters=32,
                  skip_reduction=False, filter_multiplier=2, include_top=include_top, 
                  weights=weights, input_tensor=input_tensor, pooling=pooling,
                  num_classes=num_classes, default_size=224, **kwargs)


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
