from __future__ import division

from keras.layers.convolutional import Convolution2D, AtrousConvolution2D, Conv2DTranspose, Conv2D, UpSampling2D
import keras.backend as K
import numpy as np
from dcn_vgg import dcn_vgg
from dcn_resnet import dcn_resnet, identity_block, conv_block
from gaussian_prior import LearningPrior
from attentive_convlstm import AttentiveConvLSTM
from config import *
import tensorflow as tf
from keras import losses
from keras.layers import ZeroPadding2D, AveragePooling2D, Flatten
from keras.layers import Input, Activation
from keras.layers.merge import add
from keras.layers import BatchNormalization, Concatenate
from keras.layers import Lambda, MaxPooling2D, Dropout, Reshape, Dense, RepeatVector
from keras.layers.convolutional_recurrent import ConvLSTM2D
import keras_frcnn.resnet as resnet
from keras.initializers import RandomUniform

_EPSILON = 10e-8



def repeat(x):
    return K.reshape(K.repeat(K.batch_flatten(x), nb_timestep), (b_s, nb_timestep, 7, 7, 512))
    # K.repeat(K.batch_flatten(x), nb_timestep)
    # return K.reshape(K.repeat(K.batch_flatten(x), nb_timestep), (nb_timestep, 224, 224, 3))


def repeat_shape(s):
    return (s[0], nb_timestep) + s[1:]
    # return (s[0], nb_timestep, s[1]*s[2]*s[3])


def Kreshape(x, shape):
    return K.reshape(x, shape)


def Kpool(x, pool_size):
    return K.pool2d(x, pool_size, pool_mode='avg')


def Kslice(x, location):
    return x[0:location, :, :, :]

# def upsampling(x):
#     return T.nnet.abstract_conv.bilinear_upsampling(input=x, ratio=upsampling_factor, num_input_channels=1, batch_size=b_s)


def upsampling_shape(s):
    return s[:2] + (s[2] * upsampling_factor, s[3] * upsampling_factor)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


# KL-Divergence Loss
def kl_divergence(y_true, y_pred):
    max_y_pred = K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[1, 2]), axis=1), y_pred.shape[1], axis=1), axis=1),
        y_pred.shape[2], axis=1)

    min_y_pred = K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.min(y_pred, axis=[1, 2]), axis=1), y_pred.shape[1], axis=1), axis=1),
        y_pred.shape[2], axis=1)

    y_pred = (y_pred-min_y_pred)/(max_y_pred - min_y_pred + K.epsilon())

    max_y_true = K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[1, 2]), axis=1), y_pred.shape[1], axis=1), axis=1),
        y_pred.shape[2], axis=1)

    sum_y_true = K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_true, axis=[1, 2]), axis=1), y_pred.shape[1], axis=1), axis=1),
        y_pred.shape[2], axis=1)
    sum_y_pred = K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_pred, axis=[1, 2]), axis=1), y_pred.shape[1], axis=1), axis=1),
        y_pred.shape[2], axis=1)

    y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')

    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())
    return K.sum(y_bool*(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon())))#


def w_binary_crossentropy(y_true, y_pred):
    return 1000 * losses.binary_crossentropy(y_true, y_pred)

def w_categorical_crossentropy(y_true, y_pred):
    return 1 * losses.categorical_crossentropy(y_true, y_pred)

def weighted_crossentropy(target, output):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        output: A tensor.
        target: A tensor with the same shape as `output`.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    pos_weight = 50
    if not False:
        # transform back to logits
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output / (1 - output))

    all_loss = tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=pos_weight)
    return K.mean(all_loss, axis=-1)


def schedule_vgg(epoch):
    lr = [1e-4, 1e-4, 1e-5, 1e-5, 1e-6,
          1e-6, 1e-7, 1e-7, 1e-8, 1e-8]
    return lr[epoch]


def schedule_resnet(epoch):
    lr = [1e-4, 1e-4, 1e-5, 1e-5, 1e-6,
          1e-6, 1e-7, 1e-7, 1e-8, 1e-8]
    return lr[epoch]


def sam_vgg(data):
    trainable = True#FalseTrue
    # conv_1
    conv_1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable)(data)
    conv_1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable)(conv_1_out)
    conv_1_out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv_1_out)

    # conv_2
    conv_2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(conv_1_out)
    conv_2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(conv_2_out)
    conv_2_out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_2_out)

    # conv_3
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(conv_2_out)
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(conv_3_out)
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(conv_3_out)
    conv_3_out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(conv_3_out)

    # conv_4
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(conv_3_out)
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(conv_4_out)
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(conv_4_out)
    conv_4_out = MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool', padding='same')(conv_4_out)

    # conv_5
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2, name='block5_conv1', trainable=trainable)(conv_4_out)
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2, name='block5_conv2', trainable=trainable)(conv_5_out)
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2, name='block5_conv3', trainable=trainable)(conv_5_out)

    # conv_5_out = Flatten()(conv_5_out)
    # conv_5_out = RepeatVector(nb_timestep)(conv_5_out)
    # conv_5_out = Reshape((nb_timestep, 28, 28, 512))(conv_5_out)
    #
    #
    # # part land output
    # part_land_outs = (ConvLSTM2D(filters=512, kernel_size=(3, 3),
    #                              padding='same', return_sequences=False, stateful=False))(conv_5_out)
    # part_land_outs = BatchNormalization()(part_land_outs)
    # part_land_outs = Activation('sigmoid')(part_land_outs)
    part_land_outs = Conv2D(8, (3, 3), padding='same', activation='sigmoid', name='part_land', trainable=trainable)(
        conv_5_out)

    # part body land output
    part_body_land_outs = Conv2D(2, (3, 3), padding='same', activation='sigmoid', name='part_body_land', trainable=trainable)(
        conv_5_out)
    # part_body_land_outs = (ConvLSTM2D(filters=2, kernel_size=(3, 3),
    #                              padding='same', activation='sigmoid', return_sequences=False, stateful=False,
    #                              name='part_body_land'))(conv_5_out)

    # full body land output
    full_body_land_outs = Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='full_body_land', trainable=trainable)(
        conv_5_out)


    # full_body_land_outs = (ConvLSTM2D(filters=1, kernel_size=(3, 3),
    #                                   padding='same', activation='sigmoid', return_sequences=False, stateful=False,
    #                                   name='full_body_land'))(conv_5_out)

    # outs = Flatten()(conv_5_out)
    # outs = RepeatVector(nb_timestep)(outs)
    # outs = Reshape((nb_timestep, 28, 28, 512))(outs)
    # attenLSTM_outs = AttentiveConvLSTM(nb_filters_in=512, nb_filters_out=512, nb_filters_att=512,
    #                          nb_cols=3, nb_rows=3)(outs)
    # attenLSTM_outs = Lambda(Kreshape, arguments={'shape': [-1, 28, 28, 512]}, output_shape=[28, 28, 512])(attenLSTM_outs)

    beta_part_body_conv = Conv2D(16, (3, 3), padding='same', activation='relu', trainable=trainable)(part_land_outs)
    beta_part_body_conv = MaxPooling2D((2, 2), strides=(2, 2))(beta_part_body_conv)
    beta_part_body_conv = Conv2D(16, (5, 5), padding='same', activation='relu', trainable=trainable)(beta_part_body_conv)
    beta_part_body_conv = Conv2D(16, (5, 5), padding='same', activation='relu', trainable=trainable)(beta_part_body_conv)
    beta_part_body_land_outs = Conv2D(2, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(
        beta_part_body_conv)
    beta_part_body_land_outs = UpSampling2D(size=(2, 2), name='beta_part_body_land')(beta_part_body_land_outs)

    beta_full_body_conv = Conv2D(8, (3, 3), padding='same', activation='relu', trainable=trainable)(part_body_land_outs)
    beta_full_body_conv = MaxPooling2D((2, 2), strides=(2, 2))(beta_full_body_conv)
    beta_full_body_conv = Conv2D(8, (5, 5), padding='same', activation='relu', trainable=trainable)(beta_full_body_conv)
    beta_full_body_conv = Conv2D(8, (5, 5), padding='same', activation='relu', trainable=trainable)(beta_full_body_conv)
    beta_full_body_land_outs = Conv2D(1, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(
        beta_full_body_conv)
    beta_full_body_land_outs = UpSampling2D(size=(2, 2), name='beta_full_body_land')(beta_full_body_land_outs)

    # gamma
    gamma_part_body_conv = Conv2D(16, (3, 3), padding='same', activation='relu', trainable=trainable)(full_body_land_outs)
    gamma_part_body_conv = MaxPooling2D((2, 2), strides=(2, 2))(gamma_part_body_conv)
    gamma_part_body_conv = Conv2D(16, (5, 5), padding='same', activation='relu', trainable=trainable)(gamma_part_body_conv)
    gamma_part_body_conv = Conv2D(16, (5, 5), padding='same', activation='relu', trainable=trainable)(gamma_part_body_conv)
    gamma_part_body_land_outs = Conv2D(2, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(
        gamma_part_body_conv)
    gamma_part_body_land_outs = UpSampling2D(size=(2, 2), name='gamma_part_body_land')(gamma_part_body_land_outs)

    # gamma
    gamma_part_conv = Conv2D(64, (3, 3), padding='same', activation='relu', trainable=trainable)(part_body_land_outs)
    gamma_part_conv = MaxPooling2D((2, 2), strides=(2, 2))(gamma_part_conv)
    gamma_part_conv = Conv2D(64, (5, 5), padding='same', activation='relu', trainable=trainable)(gamma_part_conv)
    gamma_part_conv = Conv2D(64, (5, 5), padding='same', activation='relu', trainable=trainable)(gamma_part_conv)
    gamma_part_land_outs = Conv2D(8, (3, 3), padding='same', activation='sigmoid', trainable=trainable)(
        gamma_part_conv)
    gamma_part_land_outs = UpSampling2D(size=(2, 2), name='gamma_part_land')(gamma_part_land_outs)

    part_land_outs1 = Lambda(lambda x: K.expand_dims(x[:, :, :, 0]), output_shape=(28, 28, 1))(part_land_outs)
    part_land_outs2 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1]), output_shape=(28, 28, 1))(part_land_outs)
    part_land_outs3 = Lambda(lambda x: K.expand_dims(x[:, :, :, 2]), output_shape=(28, 28, 1))(part_land_outs)
    part_land_outs4 = Lambda(lambda x: K.expand_dims(x[:, :, :, 3]), output_shape=(28, 28, 1))(part_land_outs)
    part_land_outs5 = Lambda(lambda x: K.expand_dims(x[:, :, :, 4]), output_shape=(28, 28, 1))(part_land_outs)
    part_land_outs6 = Lambda(lambda x: K.expand_dims(x[:, :, :, 5]), output_shape=(28, 28, 1))(part_land_outs)
    part_land_outs7 = Lambda(lambda x: K.expand_dims(x[:, :, :, 6]), output_shape=(28, 28, 1))(part_land_outs)
    part_land_outs8 = Lambda(lambda x: K.expand_dims(x[:, :, :, 7]), output_shape=(28, 28, 1))(part_land_outs)

    gamma_part_land_outs1 = Lambda(lambda x: K.expand_dims(x[:, :, :, 0]), output_shape=(28, 28, 1))(gamma_part_land_outs)
    gamma_part_land_outs2 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1]), output_shape=(28, 28, 1))(gamma_part_land_outs)
    gamma_part_land_outs3 = Lambda(lambda x: K.expand_dims(x[:, :, :, 2]), output_shape=(28, 28, 1))(gamma_part_land_outs)
    gamma_part_land_outs4 = Lambda(lambda x: K.expand_dims(x[:, :, :, 3]), output_shape=(28, 28, 1))(gamma_part_land_outs)
    gamma_part_land_outs5 = Lambda(lambda x: K.expand_dims(x[:, :, :, 4]), output_shape=(28, 28, 1))(gamma_part_land_outs)
    gamma_part_land_outs6 = Lambda(lambda x: K.expand_dims(x[:, :, :, 5]), output_shape=(28, 28, 1))(gamma_part_land_outs)
    gamma_part_land_outs7 = Lambda(lambda x: K.expand_dims(x[:, :, :, 6]), output_shape=(28, 28, 1))(gamma_part_land_outs)
    gamma_part_land_outs8 = Lambda(lambda x: K.expand_dims(x[:, :, :, 7]), output_shape=(28, 28, 1))(gamma_part_land_outs)


    com_part_land_outs1 = Concatenate()([part_land_outs1, gamma_part_land_outs1])
    com_part_land_outs1 = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), trainable=True)(com_part_land_outs1)
    com_part_land_outs2 = Concatenate()([part_land_outs2, gamma_part_land_outs2])
    com_part_land_outs2 = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), trainable=True)(com_part_land_outs2)
    com_part_land_outs3 = Concatenate()([part_land_outs3, gamma_part_land_outs3])
    com_part_land_outs3 = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), trainable=True)(com_part_land_outs3)
    com_part_land_outs4 = Concatenate()([part_land_outs4, gamma_part_land_outs4])
    com_part_land_outs4 = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), trainable=True)(com_part_land_outs4)
    com_part_land_outs5 = Concatenate()([part_land_outs5, gamma_part_land_outs5])
    com_part_land_outs5 = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), trainable=True)(com_part_land_outs5)
    com_part_land_outs6 = Concatenate()([part_land_outs6, gamma_part_land_outs6])
    com_part_land_outs6 = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), trainable=True)(com_part_land_outs6)
    com_part_land_outs7 = Concatenate()([part_land_outs7, gamma_part_land_outs7])
    com_part_land_outs7 = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), trainable=True)(com_part_land_outs7)
    com_part_land_outs8 = Concatenate()([part_land_outs8, gamma_part_land_outs8])
    com_part_land_outs8 = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), trainable=True)(com_part_land_outs8)

    com_part_land_outs = Concatenate(name='com_part_land')([com_part_land_outs1, com_part_land_outs2, com_part_land_outs3, com_part_land_outs4,
                                        com_part_land_outs5, com_part_land_outs6, com_part_land_outs7, com_part_land_outs8])

    part_body_land_outs1 = Lambda(lambda x: K.expand_dims(x[:, :, :, 0]), output_shape=(28, 28, 1))(part_body_land_outs)
    part_body_land_outs2 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1]), output_shape=(28, 28, 1))(part_body_land_outs)
    beta_part_body_land_outs1 = Lambda(lambda x: K.expand_dims(x[:, :, :, 0]), output_shape=(28, 28, 1))(beta_part_body_land_outs)
    beta_part_body_land_outs2 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1]), output_shape=(28, 28, 1))(beta_part_body_land_outs)
    gamma_part_body_land_outs1 = Lambda(lambda x: K.expand_dims(x[:, :, :, 0]), output_shape=(28, 28, 1))(gamma_part_body_land_outs)
    gamma_part_body_land_outs2 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1]), output_shape=(28, 28, 1))(gamma_part_body_land_outs)

    com_part_body_land_outs1 = Concatenate()([part_body_land_outs1, beta_part_body_land_outs1, gamma_part_body_land_outs1])
    com_part_body_land_outs1 = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), trainable=True)(com_part_body_land_outs1)
    com_part_body_land_outs2 = Concatenate()([part_body_land_outs2, beta_part_body_land_outs2, gamma_part_body_land_outs2])
    com_part_body_land_outs2 = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), trainable=True)(com_part_body_land_outs2)

    com_part_body_land_outs = Concatenate(name='com_part_body_land')([com_part_body_land_outs1, com_part_body_land_outs2])

    com_full_body_land_outs = Concatenate()([full_body_land_outs, beta_full_body_land_outs])
    com_full_body_land_outs = Conv2D(1, (1, 1), kernel_initializer=RandomUniform(minval=0, maxval=1, seed=None), name='com_full_body_land', trainable=True)(com_full_body_land_outs)

    return [part_land_outs, part_body_land_outs, full_body_land_outs,
            beta_part_body_land_outs, beta_full_body_land_outs,
            gamma_part_body_land_outs, gamma_part_land_outs,
            com_part_land_outs, com_part_body_land_outs, com_full_body_land_outs]#


def sam_resnet(data):

    # dcn = dcn_resnet(input_tensor=data, trainable=True)
    bn_axis = 3
    trainable = True#
    # conv_1
    conv_1_out = ZeroPadding2D((3, 3), batch_size=1)(data)
    conv_1_out = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(conv_1_out)
    conv_1_out = BatchNormalization(axis=bn_axis, name='bn_conv1', trainable=trainable)(conv_1_out)
    conv_1_out_b = Activation('relu')(conv_1_out)
    conv_1_out = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv_1_out_b)

    # conv_2
    conv_2_out = conv_block(conv_1_out, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
    conv_2_out = identity_block(conv_2_out, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
    conv_2_out = identity_block(conv_2_out, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

    # conv_3
    conv_3_out = conv_block(conv_2_out, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2), trainable=trainable)
    conv_3_out = identity_block(conv_3_out, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
    conv_3_out = identity_block(conv_3_out, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
    conv_3_out = identity_block(conv_3_out, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)

    # conv_4
    conv_4_out = conv_block(conv_3_out, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
    conv_4_out = identity_block(conv_4_out, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
    conv_4_out = identity_block(conv_4_out, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
    conv_4_out = identity_block(conv_4_out, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
    conv_4_out = identity_block(conv_4_out, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
    conv_4_out = identity_block(conv_4_out, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)

    # conv_5
    conv_5_out = conv_block(conv_4_out, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1), trainable=trainable)  #
    conv_5_out = identity_block(conv_5_out, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    conv_5_out = identity_block(conv_5_out, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    #
    # processing Resnet output
    resnet_outs = Conv2D(512, (3, 3), padding='same', activation='relu', name='resnet_out', trainable=trainable)(conv_5_out)
    resnet_outs = Flatten()(resnet_outs)
    resnet_outs = RepeatVector(nb_timestep)(resnet_outs)
    resnet_outs = Reshape((nb_timestep, 14, 14, 512))(resnet_outs)

    # Attentive Convolutional LSTM
    convLSTM_outs = AttentiveConvLSTM(nb_filters_in=512, nb_filters_out=512, nb_filters_att=512,
                              nb_cols=3, nb_rows=3, name='attenconvsltm')(resnet_outs)#, trainable=True
    convLSTM_outs = Lambda(Kreshape, arguments={'shape': [-1, 14, 14, 512]}, output_shape=[14, 14, 512])(convLSTM_outs)
    # final land output
    land_outs = Conv2D(8, (1, 1), padding='same', activation='sigmoid', name='land_con5', trainable=True)(convLSTM_outs)
    # upsamping land output for 3rd block
    conv_3_out = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_3_out', trainable=True)(conv_3_out)
    up3_land_outs = UpSampling2D(size=(2, 2))(land_outs)
    up3_land_outs = Concatenate()([conv_3_out, up3_land_outs])
    up3_land_outs = Flatten()(up3_land_outs)
    up3_land_outs = RepeatVector(nb_timestep)(up3_land_outs)
    up3_land_outs = Reshape((nb_timestep, 28, 28, 72))(up3_land_outs)
    up3_land_outs = (ConvLSTM2D(filters=8, kernel_size=(3, 3),
                       padding='same', activation='sigmoid', return_sequences=False, stateful=False, name='land_con3'))(up3_land_outs)
    # # upsamping land output for 2nd block
    conv_2_out = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_2_out', trainable=True)(conv_2_out)
    up2_land_outs = UpSampling2D(size=(2, 2))(up3_land_outs)
    up2_land_outs = Concatenate()([conv_2_out, up2_land_outs])
    up2_land_outs = Flatten()(up2_land_outs)
    up2_land_outs = RepeatVector(nb_timestep)(up2_land_outs)
    up2_land_outs = Reshape((nb_timestep, 56, 56, 72))(up2_land_outs)
    up2_land_outs = (ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                padding='same', activation='sigmoid', return_sequences=False, stateful=False,
                                name='land_con2'))(up2_land_outs)

    # # # upsamping land output for 1st block
    up1_land_outs = UpSampling2D(size=(2, 2))(up2_land_outs)
    up1_land_outs = Concatenate()([conv_1_out_b, up1_land_outs])
    up1_land_outs = Flatten()(up1_land_outs)
    up1_land_outs = RepeatVector(nb_timestep)(up1_land_outs)
    up1_land_outs = Reshape((nb_timestep, 112, 112, 72))(up1_land_outs)
    up1_land_outs = (ConvLSTM2D(filters=8, kernel_size=(3, 3),
                                padding='same', activation='sigmoid', return_sequences=False, stateful=False,
                                name='land_con1'))(up1_land_outs)

    # outs = Lambda(Kpool, arguments={'pool_size': (14, 14)}, output_shape=[1, 1, 512])(outs)
    # # outs = K.pool2d(outs, (7,7), pool_mode='avg')

    outs = AveragePooling2D((14, 14), name='avg_pool')(convLSTM_outs)
    outs = Flatten()(outs)
    #
    attri_outs = Dense(1000, kernel_initializer='normal', activation='sigmoid', name='attri', trainable=trainable)(outs)
    #
    cate_outs = Dense(cate_num, kernel_initializer='normal', activation='softmax', name='cate', trainable=trainable)(outs)
    #
    type_outs = Dense(type_num, kernel_initializer='normal', activation='softmax', name='type', trainable=trainable)(outs)

    # land_outs = Dense(196, kernel_initializer='normal', activation='sigmoid', name='land_all', trainable=True)(outs)
    # land_outs = Reshape((14, 14, 1))(land_outs)
    return [attri_outs, cate_outs, type_outs, land_outs, up3_land_outs, up2_land_outs, up1_land_outs]#

# def sam_resnet(data):
#     dcn = dcn_resnet(input_tensor=data)
#
#     rpn = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
#         dcn.output)
#
#     rpn_class = Conv2D(9, (1, 1), activation='sigmoid', kernel_initializer='uniform',
#                             name='rpn_out_class')(rpn)
#     rpn_regr = Conv2D(9*4, (1, 1), activation='linear', kernel_initializer='zero',
#                            name='rpn_out_regress')(rpn)
#
#     outs = resnet.conv_block(dcn.output, 3, [512, 512, 2048], stage=5, block='a', trainable=True)
#     outs = resnet.identity_block(outs, 3, [512, 512, 2048], stage=5, block='b', trainable=True)
#     outs = resnet.identity_block(outs, 3, [512, 512, 2048], stage=5, block='c', trainable=True)
#     outs = Conv2D(512, (3, 3), padding='same', activation='relu')(outs)
#
#     outs = Flatten()(outs)
#     outs = RepeatVector(nb_timestep)(outs)
#     outs = Reshape((nb_timestep, 7, 7, 512))(outs)
#     outs = AttentiveConvLSTM(nb_filters_in=512, nb_filters_out=512, nb_filters_att=512,
#                              nb_cols=3, nb_rows=3)(outs)
#     outs = Lambda(Kpool, arguments={'pool_size': (7, 7)}, output_shape=[1, 1, 512])(outs)
#     outs = Flatten()(outs)
#
#     attri_outs = Dense(1000, kernel_initializer='normal', activation='sigmoid')(outs)
#     cate_outs = Dense(cate_num, kernel_initializer='normal', activation='softmax')(outs)
#
#     return [attri_outs, cate_outs, rpn_class, rpn_regr]






