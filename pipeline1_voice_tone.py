"""
Pipeline 1: Voice & Tone Analysis
Analyzes vocal characteristics, emotional tone, and delivery quality.
"""

import librosa
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class VoiceToneAnalyzer:
    """Analyzes voice characteristics and emotional tone from audio."""
    
    def __init__(self):
        self.filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well', 'actually', 'basically']
        
    def analyze(self, audio_path: str) -> Dict:
        """
        Main analysis function that processes audio and returns comprehensive voice analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing all voice analysis metrics and scores
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        # Extract features
        features = self._extract_vocal_features(y, sr, duration)
        emotional_tone = self._detect_emotional_tone(y, sr, features)
        negative_behaviors = self._detect_negative_behaviors(y, sr, features, duration)
        delivery_score = self._calculate_delivery_score(features, emotional_tone, negative_behaviors)
        
        return {
            'vocal_features': features,
            'emotional_tone': emotional_tone,
            'negative_behaviors': negative_behaviors,
            'delivery_score': delivery_score,
            'duration_seconds': duration
        }
    
    def _extract_vocal_features(self, y: np.ndarray, sr: int, duration: float) -> Dict:
        """Extract pitch, pace, volume variation, and pause information."""
        # Pitch analysis (F0 - fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        pitch_values = np.array(pitch_values)
        
        # Pace (words per minute approximation)
        # Using onset detection to estimate speech rate
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        pace = len(onsets) / duration * 60 if duration > 0 else 0
        
        # Volume variation (RMS energy)
        rms = librosa.feature.rms(y=y)[0]
        volume_variation = np.std(rms) / (np.mean(rms) + 1e-6)
        avg_volume = np.mean(rms)
        
        # Pause detection (silence detection)
        frame_length = 2048
        hop_length = 512
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        silence_threshold = np.percentile(rms_frames, 20)
        silence_frames = rms_frames < silence_threshold
        pause_count = np.sum(np.diff(silence_frames.astype(int)) == 1)
        pause_ratio = np.sum(silence_frames) / len(silence_frames) if len(silence_frames) > 0 else 0
        
        # Pitch statistics
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        pitch_range = np.max(pitch_values) - np.min(pitch_values) if len(pitch_values) > 0 else 0
        
        return {
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'pitch_range': float(pitch_range),
            'pace_onsets_per_min': float(pace),
            'volume_variation': float(volume_variation),
            'avg_volume': float(avg_volume),
            'pause_count': int(pause_count),
            'pause_ratio': float(pause_ratio)
        }
    
    def _detect_emotional_tone(self, y: np.ndarray, sr: int, features: Dict) -> Dict:
        """Detect emotional tone: confidence, nervousness, enthusiasm."""
        # Extract MFCC features for emotion analysis
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Confidence indicators
        # Higher pitch variation and volume variation suggest confidence
        pitch_variation_score = min(features['pitch_std'] / 50.0, 1.0) if features['pitch_std'] > 0 else 0
        volume_variation_score = min(features['volume_variation'] * 2, 1.0)
        confidence_score = (pitch_variation_score + volume_variation_score) / 2
        
        # Nervousness indicators
        # High pause ratio, low pitch variation, monotonicity
        pause_penalty = min(features['pause_ratio'] * 2, 1.0)
        monotonicity = 1.0 - min(features['pitch_std'] / 30.0, 1.0) if features['pitch_std'] > 0 else 1.0
        nervousness_score = (pause_penalty + monotonicity) / 2
        
        # Enthusiasm indicators
        # Higher average pitch, more volume variation, faster pace
        pitch_enthusiasm = min(features['pitch_mean'] / 300.0, 1.0) if features['pitch_mean'] > 0 else 0
        pace_enthusiasm = min(features['pace_onsets_per_min'] / 200.0, 1.0)
        enthusiasm_score = (pitch_enthusiasm + volume_variation_score + pace_enthusiasm) / 3
        
        return {
            'confidence': float(confidence_score),
            'nervousness': float(nervousness_score),
            'enthusiasm': float(enthusiasm_score)
        }
    
    def _detect_negative_behaviors(self, y: np.ndarray, sr: int, features: Dict, duration: float) -> Dict:
        """Detect filler words, hesitation, and monotonicity."""
        # Monotonicity (low pitch variation)
        pitch_cv = features['pitch_std'] / (features['pitch_mean'] + 1e-6)
        monotonicity_score = max(0, 1.0 - pitch_cv * 10)
        
        # Hesitation index (based on pauses and low energy periods)
        hesitation_score = features['pause_ratio'] * 0.7 + (1.0 - features['volume_variation']) * 0.3
        
        # Filler word detection would require transcription
        # For now, we estimate based on pause patterns and speech disfluencies
        # This is a placeholder - in production, use ASR transcript
        estimated_filler_frequency = hesitation_score * 0.5
        
        return {
            'monotonicity': float(monotonicity_score),
            'hesitation_index': float(hesitation_score),
            'estimated_filler_frequency': float(estimated_filler_frequency)
        }
    
    def _calculate_delivery_score(self, features: Dict, emotional_tone: Dict, negative_behaviors: Dict) -> float:
        """Calculate overall delivery score (0-100) based on clarity, energy, and confidence."""
        # Clarity score (inverse of hesitation and monotonicity)
        clarity = (1.0 - negative_behaviors['hesitation_index']) * 0.5 + \
                  (1.0 - negative_behaviors['monotonicity']) * 0.5
        
        # Energy score (based on volume variation, pace, enthusiasm)
        energy = emotional_tone['enthusiasm'] * 0.4 + \
                 min(features['volume_variation'] * 2, 1.0) * 0.3 + \
                 min(features['pace_onsets_per_min'] / 150.0, 1.0) * 0.3
        
        # Confidence score
        confidence = emotional_tone['confidence'] * 0.6 + \
                    (1.0 - emotional_tone['nervousness']) * 0.4
        
        # Weighted final score
        delivery_score = (clarity * 0.35 + energy * 0.35 + confidence * 0.30) * 100
        
        return round(max(0, min(100, delivery_score)), 2)
