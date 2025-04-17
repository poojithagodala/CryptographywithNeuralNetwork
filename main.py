from data_generator import generate_random_text, generate_dummy_images
from encoder import build_encoder
from decoder import build_decoder
import numpy as np

# Generate inputs
text = generate_random_text()
image = generate_dummy_images()

# Preprocess text to numerical (simplified)
text_numerical = np.array([[ord(c) % 100 for c in text[:400]]])  # max length = 400
image = np.expand_dims(image, axis=0)  # batch

# Load models
encoder = build_encoder()
decoder = build_decoder()

# Encode
encoded_image = encoder.predict([image, text_numerical])

# Decode
decoded = decoder.predict(encoded_image)

print("Decoded output shape:", decoded.shape)