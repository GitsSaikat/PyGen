import librosa
import noisereduce as nr
from scipy.signal import savgol_filter
import numpy as np
from typing import Tuple, Optional
import logging

class AudioPreprocessor:
    """Enhanced audio preprocessing with security and validation checks"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
    def load_audiofile(self, filename: str) -> Tuple[np.ndarray, int]:
        """Securely load and validate audio file"""
        try:
            # Validate file extension
            if not filename.lower().endswith(('.wav', '.mp3', '.flac')):
                raise ValueError("Unsupported audio format. Use WAV, MP3, or FLAC.")
                
            signal, sample_rate = librosa.load(filename, sr=self.sample_rate)
            
            # Validate audio data
            if len(signal) == 0:
                raise ValueError("Empty audio file detected")
                
            return signal, sample_rate
            
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}")
            raise

    def denoise_audio(self, signal: np.ndarray, 
                     noise_threshold: float = 0.001,
                     stationary: bool = True) -> np.ndarray:
        """Enhanced denoising with multiple algorithms"""
        try:
            # Stationary noise reduction
            if stationary:
                signal = nr.reduce_noise(y=signal, sr=self.sample_rate)
            
            # Non-stationary noise reduction
            else:
                signal = nr.reduce_noise(y=signal, sr=self.sample_rate,
                                      stationary=False,
                                      prop_decrease=0.75)
            
            return signal
        except Exception as e:
            self.logger.error(f"Error in denoising: {str(e)}")
            raise

    def normalize_audio(self, signal: np.ndarray, 
                       target_db: float = -20.0) -> np.ndarray:
        """Normalize audio with target dB level"""
        try:
            # Peak normalization
            peak = np.abs(signal).max()
            if peak > 0:
                normalized = signal / peak
                
            # RMS normalization to target dB
            rms = np.sqrt(np.mean(normalized**2))
            target_rms = 10**(target_db/20)
            normalized *= (target_rms / rms)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error in normalization: {str(e)}")
            raise

    def preprocess_audio(self, signal: np.ndarray,
                        apply_filters: bool = True) -> np.ndarray:
        """Enhanced preprocessing pipeline with additional filters"""
        try:
            # 1. Noise reduction
            signal = self.denoise_audio(signal)
            
            # 2. Apply additional filters
            if apply_filters:
                # Savitzky-Golay filter for smoothing
                signal = savgol_filter(signal, 51, 3)
                
                # High-pass filter to remove DC offset
                signal = librosa.effects.preemphasis(signal)
                
                # Trim silence
                signal, _ = librosa.effects.trim(signal, top_db=20)
            
            # 3. Normalize audio level
            signal = self.normalize_audio(signal)
            
            # 4. Validate output
            if np.isnan(signal).any():
                raise ValueError("Invalid values detected in processed audio")
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
