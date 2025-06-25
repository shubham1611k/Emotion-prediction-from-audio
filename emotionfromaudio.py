import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
import librosa
import numpy as np
import os
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from moviepy import VideoFileClip
import glob
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import torchaudio.transforms as T

def extract_audio_from_video(video_path, audio_output_path):
    if not os.path.exists(audio_output_path):
        os.makedirs(audio_output_path)
    for file in os.listdir(video_path):
        if file.endswith(".mp4"):
            file_path = os.path.join(video_path, file)
            audio_file = os.path.join(audio_output_path, file.replace(".mp4", ".wav"))
            video = VideoFileClip(file_path)
            video.audio.write_audiofile(audio_file)

#extract_audio_from_video("./data/audiotraindata", "audio")

def get_emotion_label(file):
    code = file.split("-")[2]
    emotion_map = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }
    return emotion_map.get(code, "unknown")

class EmotionDataset(Dataset):
    def __init__(self, audio_dir, max_len=862, sample_rate=22050, n_mfcc=13, specaugment=False):
        self.audio_files = []
        self.labels = []
        self.label_map = {
            'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
            'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
        }
        self.max_len = max_len
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.specaugment = specaugment
        self.spec_augment = nn.Sequential(
            T.FrequencyMasking(freq_mask_param=8),
            T.TimeMasking(time_mask_param=15)
        ) if specaugment else None
    
        for file in os.listdir(audio_dir):
            if file.endswith(".wav"):
                label = get_emotion_label(file)
                if label != "unknown":
                    self.audio_files.append(os.path.join(audio_dir, file))
                    self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.audio_files)

    def extract_features(self, audio, sr):
        """Enhanced feature extraction with more robust features"""
        # Basic MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=512, n_fft=2048)
        
        # Spectral features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=512)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        
        # Rhythm and energy features
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=512)
        rms = librosa.feature.rms(y=audio, hop_length=512)
        
        # Combine all features
        features = np.vstack([mfccs, chroma, contrast, tonnetz, zcr, rms])
        return features

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio with better preprocessing
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Extract enhanced features
        features = self.extract_features(audio, sr)

        # Padding/truncation
        if features.shape[1] < self.max_len:
            pad_width = self.max_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :self.max_len]
        
        features = torch.tensor(features, dtype=torch.float32)
        
        # Apply SpecAugment
        if self.specaugment:
            features = self.spec_augment(features)

        # Robust normalization
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True)
        features = (features - mean) / (std + 1e-8)
        
        return features, self.labels[idx]

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return F.gelu(self.conv_block(x) + self.shortcut(x))

class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, num_classes=8, dropout=0.3):
        super().__init__()

        # Enhanced feature extraction with residual blocks
        self.feature_extractor = nn.Sequential(
            ResidualBlock1D(input_dim, 128, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock1D(128, 256, dropout=dropout),
            nn.MaxPool1d(2),
            ResidualBlock1D(256, 512, dropout=dropout),
        )

        # Improved BiLSTM
        self.bi_lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Multi-head attention with more heads
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=2*hidden_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )

        # Enhanced classifier with batch normalization
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
        # CNN feature extraction
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)
        
        # BiLSTM
        lstm_out, _ = self.bi_lstm(x)
        
        # Self-attention
        attn_output, attn_weights = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling + max pooling
        avg_pool = torch.mean(attn_output, dim=1)
        max_pool, _ = torch.max(attn_output, dim=1)
        context = avg_pool + max_pool
        
        return self.classifier(context)

def train_model(epochs=300, lr=1e-4, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data with train/val split
    root_dir = "./audio"
    full_dataset = EmotionDataset(root_dir, specaugment=True, sample_rate=22050)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create validation dataset 
    val_dataset_clean = EmotionDataset(root_dir, specaugment=False, sample_rate=22050)
    val_indices = val_dataset.indices
    val_subset = torch.utils.data.Subset(val_dataset_clean, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = CNNBiLSTMAttention(input_dim=40).to(device)
    model.load_state_dict(torch.load("best_emotion_model.pth", map_location=device))
    
    # Loss and optimizer with scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
        
        val_acc = 100. * val_correct / val_total
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            torch.save(model.state_dict(), "emotion_model.pth")
            print("-" * 50)
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_emotion_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    return model

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    roots = ["./data/audiotestdata1", "./data/audiotestdata2"]
    all_datasets = []

    for root in roots:
        for subfolder in os.listdir(root):
            subfolder_path = os.path.join(root, subfolder)
            if os.path.isdir(subfolder_path):
                dataset = EmotionDataset(subfolder_path, sample_rate=22050)
                all_datasets.append(dataset)

    test_dataset = ConcatDataset(all_datasets)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

    # Load best model
    model = CNNBiLSTMAttention(input_dim=40).to(device)
    model.load_state_dict(torch.load("best_emotion_model.pth", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)

    # Evaluation
    print("\nTest Results:")
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Usage
if __name__ == "__main__":
    model = train_model(epochs=250, lr=2e-4, batch_size=64)
    evaluate_model()