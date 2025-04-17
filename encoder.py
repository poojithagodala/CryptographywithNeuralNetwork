import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Reshape, Conv2D, Concatenate
from tensorflow.keras.models import Model

def build_encoder(image_shape=(200, 200, 1), vocab_size=1000, embedding_dim=200):
    text_input = Input(shape=(None,), name='text_input')
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    reshaped_text = Reshape((200, 200, 1))(embedding)

    image_input = Input(shape=image_shape, name='image_input')
    conv_img = Conv2D(20, (1, 1), activation='relu')(image_input)

    merged = Concatenate(axis=-1)([conv_img, reshaped_text])
    encoded_output = Conv2D(3, (1, 1), activation='sigmoid', name='encoded_output')(merged)

    return Model(inputs=[image_input, text_input], outputs=encoded_output, name="Encoder")