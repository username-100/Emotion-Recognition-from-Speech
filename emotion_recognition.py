import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sounddevice as sd
import scipy.io.wavfile as wavfile

# Function to extract audio features
def extract_features(audio, sample_rate):
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)
    
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_processed = np.mean(mel.T, axis=0)
    
    # Combine features
    features = np.hstack([mfccs_processed, chroma_processed, mel_processed])
    return features

# Simulated pre-trained model (replace with actual trained model)
def load_model():
    model = RandomForestClassifier(n_estimators=100)
    return model

# Record audio from microphone
def record_audio(duration=5, sample_rate=44100):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    audio = audio.flatten()  # Convert to 1D array
    return audio, sample_rate

# Predict emotion
def predict_emotion(audio, sample_rate, model):
    features = extract_features(audio, sample_rate)
    features = features.reshape(1, -1)  # Reshape for model input
    
    #replace with actual model prediction
    emotions = ['happy', 'sad', 'angry', 'neutral']
    prediction = np.random.choice(emotions)  # Simulated random prediction
    return prediction

def main():
    # Load model
    model = load_model()
    
    # Record audio
    audio, sample_rate = record_audio()
    
    # Save audio to file (optional)
    wavfile.write('recorded_audio.wav', sample_rate, audio)
    
    # Predict emotion
    emotion = predict_emotion(audio, sample_rate, model)
    print(f"Predicted emotion: {emotion}")

if __name__ == "__main__":
    main()