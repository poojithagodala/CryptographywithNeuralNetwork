from encoder import build_encoder
from decoder import build_decoder

encoder_model = build_encoder()
decoder_model = build_decoder()

encoder_model.summary()
decoder_model.summary()