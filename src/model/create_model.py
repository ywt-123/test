from typing import Union

import keras
import keras.activations
from keras import layers


def make_model(inputs_size, activation="relu") -> keras.Model:
    labels = ["MCD", "sp", "t2m", "total_precipitation", "u10", "v10", "DEM"]
    labels_length = len(labels)
    inputs_list = []
    outputs_list = [i for i in range(labels_length)]
    outputs_list_1 = [i for i in range(labels_length)]
    outputs_list_2 = [i for i in range(labels_length)]
    outputs_list_3 = [i for i in range(labels_length)]
    outputs_list_4 = [i for i in range(labels_length)]

    for label in labels:
        inputs_list.append(layers.Input(shape=inputs_size[0], name=label))
        
    landmask_inputs = layers.Input(shape=inputs_size[0], name="landmask")

    # (256, 256) -> (128, 128)
    for index in range(len(outputs_list)):
        outputs_list[index] = layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation=activation)(inputs_list[index])

    # Normalization
    for index in range(len(outputs_list)):
        outputs_list[index] = layers.BatchNormalization()(outputs_list[index])

    # (128, 128) -> (64, 64)
    for index in range(len(outputs_list)):
        outputs_list_1[index] = layers.Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), activation=activation)(outputs_list[index])

    # Normalization
    for index in range(len(outputs_list)):
        outputs_list_1[index] = layers.BatchNormalization()(outputs_list_1[index])

    # (64, 64) -> (32, 32)
    for index in range(len(outputs_list)):
        outputs_list_2[index] = layers.Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), activation=activation)(outputs_list_1[index])

    # Normalization
    for index in range(len(outputs_list)):
        outputs_list_2[index] = layers.BatchNormalization()(outputs_list_2[index])

    # (32, 32) -> (16, 16)
    for index in range(len(outputs_list)):
        outputs_list_3[index] = layers.Conv2D(filters=512, kernel_size=(2, 2), strides=(2, 2), activation=activation)(outputs_list_2[index])

    # Normalization
    for index in range(len(outputs_list)):
        outputs_list_3[index] = layers.BatchNormalization()(outputs_list_3[index])

    # (16, 16) -> (8, 8)
    for index in range(len(outputs_list)):
        outputs_list_4[index] = layers.Conv2D(filters=1024, kernel_size=(2, 2), strides=(2, 2), activation=activation)(outputs_list_3[index])

    # Normalization
    for index in range(len(outputs_list)):
        outputs_list_4[index] = layers.BatchNormalization()(outputs_list_4[index])

    outputs_1 = layers.Concatenate()(outputs_list_4)

    # (8, 8) -> (8, 8)
    outputs_1 = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation=activation)(outputs_1)
    outputs_1 = layers.BatchNormalization()(outputs_1)

    # (8, 8) -> (16, 16)
    outputs_1 = layers.UpSampling2D()(outputs_1)
    # (16, 16) -> (16, 16)
    outputs_1 = layers.Concatenate()([outputs_1] + outputs_list_3)
    # (16, 16) -> (16, 16)
    outputs_1 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=activation)(outputs_1)
    outputs_1 = layers.BatchNormalization()(outputs_1)

    # (16, 16) -> (32, 32)
    outputs_1 = layers.UpSampling2D()(outputs_1)
    # (32, 32) -> (32, 32)
    outputs_1 = layers.Concatenate()([outputs_1] + outputs_list_2)
    # (32, 32) -> (32, 32)
    outputs_1 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=activation)(outputs_1)
    outputs_1 = layers.BatchNormalization()(outputs_1)

    # (32, 32) -> (64, 64)
    outputs_1 = layers.UpSampling2D()(outputs_1)
    # (64, 64) -> (64, 64)
    outputs_1 = layers.Concatenate()([outputs_1] + outputs_list_1)
    # (64, 64) -> (64, 64)
    outputs_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activation)(outputs_1)
    outputs_1 = layers.BatchNormalization()(outputs_1)

    # (64, 64) -> (128, 128)
    outputs_1 = layers.UpSampling2D()(outputs_1)
    # (128, 128) -> (128, 128)
    outputs_1 = layers.Concatenate()([outputs_1] + outputs_list)
    # (128, 128) -> (128, 128)
    outputs_1 = layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation=activation)(outputs_1)
    outputs_1 = layers.BatchNormalization()(outputs_1)

    # (128, 128) -> (256, 256)
    outputs_1 = layers.UpSampling2D()(outputs_1)
    outputs_1 = layers.Concatenate()([outputs_1, landmask_inputs])
    outputs_1 = layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same")(outputs_1)

    outputs_1 = layers.Reshape((256, 256))(outputs_1)

    return keras.Model(inputs_list + [landmask_inputs], outputs_1)


__all__ = ['make_model']
