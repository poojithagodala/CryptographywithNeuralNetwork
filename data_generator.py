import numpy as np
import string
import random

def generate_random_text(length=100):
    characters = string.ascii_letters + string.digits + string.punctuation + ' '
    return ''.join(random.choices(characters, k=length))

def generate_dummy_images(image_shape=(200, 200)):
    return np.random.randint(0, 256, (*image_shape, 1), dtype=np.uint8)