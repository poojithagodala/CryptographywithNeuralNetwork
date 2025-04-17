import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, TimeDistributed, Dense
from tensorflow.keras.models import Model

def build_decoder(image_shape=(200, 200, 3)):
    encoded_input = Input(shape=image_shape, name='encoded_input')
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded_input)
    x = TimeDistributed(Dense(1))(x)
    return Model(inputs=encoded_input, outputs=x, name="Decoder")