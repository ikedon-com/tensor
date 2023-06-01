# import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
# from keras.losses import SparseCategoricalCrossentropy
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras.losses import CategoricalCrossentropy
# from keras.metrics import MeanIoU
# # U-Netモデルの定義

from keras.models import Model
from keras.layers import Input, MaxPooling2D, Concatenate, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras import regularizers
from keras.optimizers import *

def create_conv(input, filters, l2_reg, name):
    x = Conv2D(filters=filters,
               kernel_size=3,               # 論文の指定通り
               activation='relu',           # 論文の指定通り
               padding='same',              # sameにすることでConcatする際にContracting側の出力のCropが不要になる
               kernel_regularizer=regularizers.l2(l2_reg),
               name=name)(input)
    x = BatchNormalization()(x)
    return x


def create_trans(input, filters, l2_reg, name):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=2,      # 論文の指定通り
                        strides=2,          # このストライドにより出力サイズが入力の2倍に拡大されている
                        activation='relu',  # 論文の指定通り
                        padding='same',     # Concat時のCrop処理回避のため
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name=name)(input)
    x = BatchNormalization()(x)
    return x

def unet():
    l2_reg = 0.0001

    input = Input((None,None,3))

    conv1_1 = create_conv(input, filters=64, l2_reg=l2_reg, name='conv1c_1')
    conv1_2 = create_conv(conv1_1, filters=64, l2_reg=l2_reg, name='conv1c_2')
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')(conv1_2)

    conv2_1 = create_conv(pool1, filters=128, l2_reg=l2_reg, name='conv2c_1')
    conv2_2 = create_conv(conv2_1, filters=128, l2_reg=l2_reg, name='conv2c_2')
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2')(conv2_2)

    conv3_1 = create_conv(pool2, filters=256, l2_reg=l2_reg, name='conv3c_1')
    conv3_2 = create_conv(conv3_1, filters=256, l2_reg=l2_reg, name='conv3c_2')
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool3')(conv3_2)

    conv4_1 = create_conv(pool3, filters=512, l2_reg=l2_reg, name='conv4c_1')
    conv4_2 = create_conv(conv4_1, filters=512, l2_reg=l2_reg, name='conv4c_2')
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool4')(conv4_2)

    conv5_1 = create_conv(pool4, filters=1024, l2_reg=l2_reg, name='conv5m_1')
    conv5_2 = create_conv(conv5_1, filters=1024, l2_reg=l2_reg, name='conv5m_2')
    trans1 = create_trans(conv5_2, filters=512, l2_reg=l2_reg, name='trans1')
    concat1 = Concatenate(name='concat1')([trans1, conv4_2])

    conv6_1 = create_conv(concat1, filters=512, l2_reg=l2_reg, name='conv6e_1')
    conv6_2 = create_conv(conv6_1, filters=512, l2_reg=l2_reg, name='conv6e_2')
    trans2 = create_trans(conv6_2, filters=256, l2_reg=l2_reg, name='trans2')
    concat2 = Concatenate(name='concat2')([trans2, conv3_2])

    conv7_1 = create_conv(concat2, filters=256, l2_reg=l2_reg, name='conv7e_1')
    conv7_2 = create_conv(conv7_1, filters=256, l2_reg=l2_reg, name='conv7e_2')
    trans3 = create_trans(conv7_2, filters=128, l2_reg=l2_reg, name='trans3')
    concat3 = Concatenate(name='concat3')([trans3, conv2_2])

    conv8_1 = create_conv(concat3, filters=128, l2_reg=l2_reg, name='conv8e_1')
    conv8_2 = create_conv(conv8_1, filters=128, l2_reg=l2_reg, name='conv8e_2')
    trans4 = create_trans(conv8_2, filters=64, l2_reg=l2_reg, name='trans4')
    concat4 = Concatenate(name='concat4')([trans4, conv1_2])

    conv9_1 = create_conv(concat4, filters=64, l2_reg=l2_reg, name='conv9e_1')
    conv9_2 = create_conv(conv9_1, filters=64, l2_reg=l2_reg, name='conv9e_2')

    output = Conv2D(filters=3,                     # VOCのカテゴリ数22
                    kernel_size=1,                  # 論文の指定通り
                    activation='softmax',           # 多クラス単一ラベル分類なのでsoftmaxを使う
                    name='output')(conv9_2)

    model = Model(input, output)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',  # 多クラス単一ラベル分類
                  metrics=['accuracy'])

    return model

loss_object = tf.keras.losses.CategoricalCrossentropy()

def softmax_cross_entropy_loss(y_true, y_pred):
    loss = loss_object(y_true, y_pred)
    return loss

def unet_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # エンコーダーブロック
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # ボトルネックブロック
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # デコーダーブロック
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    concat6 = concatenate([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(concat6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    concat7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(concat7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    concat8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(concat8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    concat9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(concat9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # 出力層
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',  # 多クラス単一ラベル分類
                  metrics=['accuracy'])

    return model