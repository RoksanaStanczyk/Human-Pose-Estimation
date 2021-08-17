import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def CNN2D():
    model = Sequential()

    model.add(layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(360, 284, 1)))
    model.add(layers.MaxPooling2D(pool_size=2))

    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))

    model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))

    model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(8))

    return model
