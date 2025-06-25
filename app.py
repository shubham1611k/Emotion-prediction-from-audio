import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import tempfile
import os
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Audio Emotion Recognition",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model Architecture (same as training code)
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.conv_block(x) + self.shortcut(x))

class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, num_classes=8, dropout=0.3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ResidualBlock1D(input_dim, 128, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock1D(128, 256, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock1D(256, 512, dropout=dropout),
        )

        self.bi_lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=2*hidden_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bi_lstm(x)
        attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        avg_pool = torch.mean(attn_output, dim=1)
        max_pool, _ = torch.max(attn_output, dim=1)
        context = avg_pool + max_pool
        return self.classifier(context)

# Feature extraction function
def extract_features(audio, sr, max_len=862, n_mfcc=13):
    """Extract features from audio file"""
    # Basic MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=512, n_fft=2048)
    # Spectral features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=512)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    
    # Rhythm and energy features
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=512)
    rms = librosa.feature.rms(y=audio, hop_length=512)
    
    # Combine all features
    features = np.vstack([mfccs, chroma, contrast, tonnetz, zcr, rms])
    
    # Padding/truncation
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_len]
    
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Normalize
    mean = features.mean(dim=2, keepdim=True)
    std = features.std(dim=2, keepdim=True)
    features = (features - mean) / (std + 1e-8)
    
    return features

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device("cpu")  # Use CPU for Streamlit deployment
    model = CNNBiLSTMAttention(input_dim=40, num_classes=8)
    
    try:
        # Try to load the model weights
        model.load_state_dict(torch.load("best_emotion_model.pth", map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model file 'emotion_model.pth' not found. Please make sure the model is trained and saved.")
        return None, device

def predict_emotion(audio_file, model, device):
    """Predict emotion from audio file"""
    # Emotion mapping
    emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load and preprocess audio
        audio, sr = librosa.load(tmp_path, sr=22050)
        audio, _ = librosa.effects.trim(audio, top_db=20)  # Trim silence
        
        # Extract features
        features = extract_features(audio, sr)
        features = features.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(features)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return emotion_labels[predicted_class], confidence, probabilities[0].cpu().numpy()
    
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None, None

def create_probability_chart(probabilities, emotion_labels):
    """Create a bar chart showing emotion probabilities"""
    df = pd.DataFrame({
        'Emotion': emotion_labels,
        'Probability': probabilities * 100
    })
    
    fig = px.bar(
        df, 
        x='Emotion', 
        y='Probability',
        title='Emotion Prediction Probabilities',
        color='Probability',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Emotions",
        yaxis_title="Probability (%)",
        showlegend=False
    )
    
    return fig

def main():
    # Title and description
    st.title("🎵 Audio Emotion Recognition")
    st.markdown("""
    This application uses a deep learning model to predict emotions from audio files.
    Upload an audio file and get real-time emotion predictions!
    """)
    
    # Sidebar
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. Upload an audio file (WAV, MP3, M4A, etc.)
    2. Wait for the model to process
    3. View the predicted emotion and confidence
    4. See probability distribution across all emotions
    """)
    
    st.sidebar.header("🎯 Supported Emotions")
    emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    for emotion in emotions:
        st.sidebar.write(f"• {emotion}")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        st.stop()
    
    # File upload
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
        help="Upload an audio file to analyze its emotional content"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**File Name:** {uploaded_file.name}")
            st.info(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            # Play audio
            st.audio(uploaded_file, format='audio/wav')
        
        # Predict button
        if st.button("🔮 Predict Emotion", type="primary"):
            with st.spinner("Analyzing audio... Please wait."):
                emotion, confidence, probabilities = predict_emotion(uploaded_file, model, device)
            
            if emotion is not None:
                # Display results
                st.success("Analysis Complete!")
                
                # Main prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Emotion", emotion)
                
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                with col3:
                    # Emoji mapping
                    emoji_map = {
                        'Neutral': '😐', 'Calm': '😌', 'Happy': '😊', 'Sad': '😢',
                        'Angry': '😠', 'Fearful': '😨', 'Disgust': '🤢', 'Surprised': '😲'
                    }
                    st.metric("Emotion", emoji_map.get(emotion, '🎭'))
                
                # Probability chart
                st.header("📊 Detailed Analysis")
                emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
                fig = create_probability_chart(probabilities, emotion_labels)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional insights
                st.header("💡 Insights")
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                
                st.write("**Top 3 Predictions:**")
                for i, idx in enumerate(top_3_indices):
                    emoji = ['🥇', '🥈', '🥉'][i]
                    st.write(f"{emoji} {emotion_labels[idx]}: {probabilities[idx]:.2%}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and PyTorch</p>
        <p><em>Upload an audio file to get started!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
