> Cryptography with Neural Networks

A project exploring secure message embedding using steganography enhanced by deep learning.

> Project Structure

- `data_generator.py`: Generates random text and dummy images.
- `encoder.py`: Neural network to embed text into images.
- `decoder.py`: Extracts text features from encoded images.
- `neural_network_driver.py`: Initializes and prints model architectures.
- `image_merge.py`: Optionally merge and display original and encoded images.
- `main.py`: Driver script to run the complete encoding/decoding pipeline.

> Requirements

- Python 3.6+
- TensorFlow 2.x
- NumPy
- OpenCV (optional for visualization)

> Run Instructions

```bash
pip install tensorflow numpy opencv-python
python main.py
```

> Sample Output

- Original Image
- Encoded Image (with embedded message)
- Decoded Text

> Concepts Used

- CNN for feature extraction
- Embedding Layers for text mapping
- TimeDistributed layers for comparison
- TensorFlow + Keras API

## Bachelors of Technology - Team Project ( Amrita, Prashath, Ragavendra, Poojitha)
