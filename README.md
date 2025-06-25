# Emotion-prediction-from-audio
# Speech Emotion Recognition with Deep Learning

A robust deep learning system for recognizing emotions from speech audio using a hybrid CNN-BiLSTM architecture with attention mechanisms. StreamLit app link- https://emotion-prediction-from-audio-fbnquh7fpsz6venbhjcwpc.streamlit.app/

## Overview

This project implements a state-of-the-art speech emotion recognition system that can classify audio samples into 8 different emotional categories. The model combines Convolutional Neural Networks (CNN) for feature extraction, Bidirectional Long Short-Term Memory (BiLSTM) networks for temporal modeling, and multi-head attention mechanisms for improved performance.

## Features

- **8 Emotion Classes**: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- **Advanced Feature Extraction**: MFCC, delta features, spectral features, rhythm and energy features
- **Data Augmentation**: Audio augmentation and SpecAugment for better generalization
- **Hybrid Architecture**: CNN + BiLSTM + Multi-head Attention
- **Training Optimizations**: Label smoothing, gradient clipping, early stopping, learning rate scheduling

## Architecture

### Model Components

1. **Feature Extractor**: Residual CNN blocks with batch normalization and GELU activation
2. **Temporal Modeling**: 3-layer bidirectional LSTM with dropout
3. **Attention Mechanism**: 8-head multi-head attention for sequence modeling
4. **Classifier**: Multi-layer perceptron with batch normalization

### Feature Engineering

The system extracts comprehensive audio features:
- **MFCC Features**: 13 Mel-frequency cepstral coefficients + delta and delta-delta
- **Spectral Features**: Chroma, spectral contrast, tonnetz
- **Rhythm Features**: Zero-crossing rate, RMS energy, spectral rolloff, spectral centroid

## Installation

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install librosa numpy scikit-learn
pip install moviepy audiomentations
pip install torchaudio
```

### Required Dependencies

```
torch>=1.9.0
librosa>=0.8.0
numpy>=1.21.0
scikit-learn>=1.0.0
moviepy>=1.0.3
audiomentations>=1.0.0
torchaudio>=0.9.0
```

## Dataset Structure

Organize your data in the following structure:

```
project/
├── audio/                    # Training audio files
│   ├── file1-emotion-code.wav
│   └── file2-emotion-code.wav
├── data/
│   ├── audiotestdata1/      # Test dataset 1
│   └── audiotestdata2/      # Test dataset 2
└── main.py                  # Main script
```

### File Naming Convention

Audio files should follow this naming pattern: `{id}-{speaker}-{emotion_code}-{intensity}.wav`

Where emotion codes are:
- `01`: Neutral
- `02`: Calm  
- `03`: Happy
- `04`: Sad
- `05`: Angry
- `06`: Fearful
- `07`: Disgust
- `08`: Surprised

## Usage

### Training the Model

```python
from main import train_model

# Train with default parameters
model = train_model(epochs=300, lr=1e-4, batch_size=16)
```

### Custom Training Parameters

```python
# Custom training configuration
model = train_model(
    epochs=200,      # Number of training epochs
    lr=5e-5,         # Learning rate
    batch_size=32    # Batch size
)
```

### Evaluating the Model

```python
from main import evaluate_model

# Evaluate on test data
evaluate_model()
```

### Video to Audio Extraction

If you have video files, extract audio first:

```python
from main import extract_audio_from_video

extract_audio_from_video("./video_path", "./audio_output_path")
```

## Model Configuration

### Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `input_dim` | 68 | Number of input features |
| `hidden_dim` | 256 | LSTM hidden dimension |
| `num_classes` | 8 | Number of emotion classes |
| `dropout` | 0.3 | Dropout rate |
| `sample_rate` | 22050 | Audio sampling rate |
| `max_len` | 862 | Maximum sequence length |
| `n_mfcc` | 13 | Number of MFCC coefficients |

### Training Configuration

- **Optimizer**: AdamW with weight decay (1e-3)
- **Loss Function**: CrossEntropyLoss with label smoothing (0.1)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Early Stopping**: Patience of 20 epochs
- **Gradient Clipping**: Max norm of 1.0

## Data Augmentation

The system employs multiple augmentation techniques:

### Audio Augmentation
- Gaussian noise addition
- Time stretching (0.9-1.1x)
- Pitch shifting (±2 semitones)
- Time shifting (±0.2 seconds)

### SpecAugment
- Frequency masking (8 frequency bins)
- Time masking (15 time steps)

## Model Performance

The model provides comprehensive evaluation metrics:

- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Detailed classification breakdown
- **Weighted F1-Score**: Overall performance metric
- **Validation Accuracy**: Real-time training monitoring

## File Structure

```
project/
├── main.py                  # Main training and evaluation script
├── best_emotion_model.pth   # Saved best model weights
├── audio/                   # Training audio directory
├── data/                    # Test data directories
└── README.md               # This file
```

## Key Functions

### `EmotionDataset`
Custom PyTorch dataset class handling:
- Audio loading and preprocessing
- Feature extraction
- Data augmentation
- Label encoding

### `CNNBiLSTMAttention`
Main model architecture with:
- Residual CNN blocks
- Bidirectional LSTM layers
- Multi-head attention
- Classification head

### `train_model()`
Complete training pipeline with:
- Data loading and splitting
- Model initialization
- Training loop with validation
- Model checkpointing

### `evaluate_model()`
Comprehensive evaluation on test data with detailed metrics.

## Tips for Better Performance

1. **Data Quality**: Ensure consistent audio quality and proper labeling
2. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
3. **Feature Engineering**: Consider additional audio features for your specific use case
4. **Data Augmentation**: Adjust augmentation parameters based on your dataset
5. **Early Stopping**: Monitor validation loss to prevent overfitting

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **Low Accuracy**: Check data quality and increase training epochs
3. **Overfitting**: Increase dropout rate or reduce model complexity
4. **Slow Training**: Use GPU acceleration and reduce number of workers if needed

### Performance Optimization

- Use GPU acceleration when available
- Optimize batch size based on available memory
- Use mixed precision training for faster training
- Consider gradient accumulation for effective larger batch sizes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.
