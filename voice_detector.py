# voice_detector.py
import numpy as np
import librosa
import soundfile as sf
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# =============================================================================
#   CROSS-PLATFORM AUDIO LOADER
# =============================================================================
def _safe_load_audio(audio_path, target_sr=22050):
    """
    Cross-platform safe audio loader.
    1) Tries soundfile (fast, reliable for WAV/FLAC).
    2) Falls back to librosa (handles MP3, etc.).
    Also normalizes to mono and resamples to target_sr.
    """
    if not os.path.exists(audio_path):
        logger.error(f"[VOICE] File not found: {audio_path}")
        return None, None

    # Try soundfile first
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1:  # stereo → mono
            y = y.mean(axis=1)
        y = y.astype(np.float32)

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Trim leading / trailing silence
        y, _ = librosa.effects.trim(y, top_db=30)
        return y, sr
    except Exception as e:
        logger.info(f"[VOICE] soundfile read failed, falling back to librosa: {e}")

    # Fallback: librosa.load
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=30)
        return y, sr
    except Exception as e:
        logger.error(f"[VOICE] librosa.load failed: {e}")
        return None, None


# =============================================================================
#   COMPREHENSIVE VOICE FEATURE EXTRACTION
# =============================================================================
def _extract_voice_features(y, sr):
    """
    Extract comprehensive voice features:
    - RMS Energy (volume/loudness)
    - RMS variability
    - Pitch (F0) using pYIN
    - Speech rate / pause ratio
    - Zero-crossing rate (voice quality)
    - Spectral centroid (brightness / tension)
    """

    min_duration_sec = 0.4  # Require at least 0.4s of audio
    if y is None or len(y) < sr * min_duration_sec:
        return None  # Audio too short / invalid

    features = {}

    # 1) RMS Energy (Volume Level)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    features["rms"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["rms_max"] = float(np.max(rms))

    # 2) Pitch Analysis (F0) using pYIN
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            sr=sr,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )
        f0_clean = f0[~np.isnan(f0)]

        if len(f0_clean) > 0:
            features["pitch"] = float(np.mean(f0_clean))
            features["pitch_std"] = float(np.std(f0_clean))
            features["pitch_min"] = float(np.min(f0_clean))
            features["pitch_max"] = float(np.max(f0_clean))
        else:
            features["pitch"] = 0.0
            features["pitch_std"] = 0.0
            features["pitch_min"] = 0.0
            features["pitch_max"] = 0.0

        # Speech rate proxy: proportion of voiced frames
        voiced_flag = voiced_flag.astype(bool)
        speech_rate = float(np.mean(voiced_flag)) if len(voiced_flag) > 0 else 0.5
        features["speech_rate"] = speech_rate
        features["pause_ratio"] = float(1.0 - speech_rate)

    except Exception as e:
        logger.info(f"[VOICE] pYIN pitch extraction failed: {e}")
        features["pitch"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_min"] = 0.0
        features["pitch_max"] = 0.0
        features["speech_rate"] = 0.5
        features["pause_ratio"] = 0.5

    # 3) Zero Crossing Rate (voice quality indicator)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    features["zcr"] = float(np.mean(zcr))

    # 4) Spectral Features (brightness / tension proxy)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    features["spectral_centroid"] = float(np.mean(spectral_centroids))

    return features


# =============================================================================
#   VOLUME LEVEL CLASSIFICATION
# =============================================================================
def _classify_volume(rms, rms_max):
    """
    Classifies volume level: Very Quiet, Quiet, Normal, Loud, Very Loud
    Returns (label, risk_score).
    """
    # Use max RMS as primary indicator, mean as backup
    if rms_max < 0.02:
        return "Very Quiet", 0.25  # can indicate low energy / blunted affect
    elif rms_max < 0.05:
        return "Quiet", 0.20
    elif rms_max < 0.12:
        return "Normal", 0.15
    elif rms_max < 0.20:
        return "Loud", 0.25
    else:
        return "Very Loud", 0.35  # high arousal / tension


# =============================================================================
#   SPEECH PACE CLASSIFICATION
# =============================================================================
def _classify_speech_pace(speech_rate, pause_ratio):
    """
    Classifies speech pace: Very Slow, Slow, Normal, Fast, Very Fast
    Uses speech_rate (proportion of voiced frames).
    Returns (label, risk_score).
    """
    if speech_rate < 0.35:
        return "Very Slow", 0.40   # strong depression / psychomotor retardation marker
    elif speech_rate < 0.50:
        return "Slow", 0.25
    elif speech_rate < 0.75:
        return "Normal", 0.15
    elif speech_rate < 0.90:
        return "Fast", 0.30        # anxiety / agitation
    else:
        return "Very Fast", 0.40   # strong anxiety / panic marker


# =============================================================================
#   PITCH-BASED STRESS DETECTION
# =============================================================================
def _analyze_pitch_stress(pitch, pitch_std, pitch_max):
    """
    Analyzes pitch characteristics for stress indicators:
    - High pitch = vocal tension (anxiety/stress)
    - High pitch variability = emotional instability
    - Very low variability but elevated pitch = possible agitation
    Returns risk_score (0.0–1.0).
    """
    stress_score = 0.0

    # High average pitch
    if pitch > 260:          # higher than typical adult average
        stress_score += 0.25
    elif pitch > 200:
        stress_score += 0.15

    # Pitch variability
    if pitch_std > 60:
        stress_score += 0.25   # high variability = emotional instability
    elif 20 < pitch_std <= 60:
        stress_score += 0.15
    elif pitch_std < 10 and pitch > 0:
        # extremely flat & non-zero: potential monotone sadness / blunting
        stress_score += 0.10

    # Very high peaks
    if pitch_max > 400:
        stress_score += 0.15

    return min(stress_score, 1.0)


# =============================================================================
#   ENERGY VARIABILITY (EMOTIONAL LABILITY)
# =============================================================================
def _energy_variation_risk(rms_std):
    """
    Large swings in energy can be a sign of agitation / emotional arousal.
    """
    if rms_std < 0.01:
        return 0.15
    if rms_std < 0.03:
        return 0.20
    if rms_std < 0.06:
        return 0.30
    return 0.40


# =============================================================================
#   MAIN VOICE ANALYSIS FUNCTION
# =============================================================================
def analyze_voice(audio_path):
    """
    Comprehensive voice stress analyzer.
    Detects:
    - Volume level (quiet, normal, loud, very loud)
    - Speech pace (very slow, slow, normal, fast, very fast)
    - Pitch-based stress indicators
    - Overall voice-based mental health risk

    Returns: {
        "stress_level": str,
        "volume_level": str,
        "speech_pace": str,
        "voice_risk": float (0.0–1.0),
        "features": dict
    }
    """
    if not os.path.exists(audio_path):
        return {
            "stress_level": "No Audio File",
            "volume_level": "N/A",
            "speech_pace": "N/A",
            "voice_risk": 0.0,
            "features": {}
        }

    # Load audio safely
    y, sr = _safe_load_audio(audio_path)
    if y is None or sr is None:
        return {
            "stress_level": "Audio Decode Error",
            "volume_level": "N/A",
            "speech_pace": "N/A",
            "voice_risk": 0.0,
            "features": {}
        }

    # Extract features
    features = _extract_voice_features(y, sr)
    if features is None:
        return {
            "stress_level": "Insufficient Audio",
            "volume_level": "N/A",
            "speech_pace": "N/A",
            "voice_risk": 0.0,
            "features": {}
        }

    # -------------------------------------------------------------------------
    # MULTI-DIMENSIONAL VOICE ANALYSIS
    # -------------------------------------------------------------------------

    # 1) Volume Analysis
    volume_level, volume_risk = _classify_volume(
        features["rms"],
        features["rms_max"]
    )

    # 2) Speech Pace Analysis
    speech_pace, pace_risk = _classify_speech_pace(
        features["speech_rate"],
        features["pause_ratio"]
    )

    # 3) Pitch-Based Stress Analysis
    pitch_risk = _analyze_pitch_stress(
        features["pitch"],
        features["pitch_std"],
        features["pitch_max"]
    )

    # 4) Energy Variability (emotional instability indicator)
    rms_variation_risk = _energy_variation_risk(features["rms_std"])

    # -------------------------------------------------------------------------
    #   SMART COMBINED VOICE RISK SCORE
    # -------------------------------------------------------------------------
    # Weight speech pace & pitch more than volume:
    voice_risk = (
        volume_risk       * 0.20 +
        pace_risk         * 0.40 +   # speech rate = strongest indicator
        pitch_risk        * 0.30 +
        rms_variation_risk * 0.10
    )

    voice_risk = max(0.0, min(1.0, voice_risk))

    # Overall qualitative stress level
    if voice_risk >= 0.70:
        stress_level = "Very High Stress"
    elif voice_risk >= 0.50:
        stress_level = "High Stress"
    elif voice_risk >= 0.30:
        stress_level = "Moderate Stress"
    else:
        stress_level = "Low Stress"

    return {
        "stress_level": stress_level,
        "volume_level": volume_level,
        "speech_pace": speech_pace,
        "voice_risk": round(float(voice_risk), 2),
        "features": {
            "rms": features["rms"],
            "rms_std": features["rms_std"],
            "rms_max": features["rms_max"],
            "pitch": features["pitch"],
            "pitch_std": features["pitch_std"],
            "pitch_min": features["pitch_min"],
            "pitch_max": features["pitch_max"],
            "speech_rate": features["speech_rate"],
            "pause_ratio": features["pause_ratio"],
            "zcr": features["zcr"],
            "spectral_centroid": features["spectral_centroid"],
            "volume_risk": volume_risk,
            "pace_risk": pace_risk,
            "pitch_risk": pitch_risk,
            "rms_variation_risk": rms_variation_risk,
        }
    }


# =============================================================================
#   LOCAL TEST MODE
# =============================================================================
if __name__ == "__main__":
    test_file = "test.wav"

    print("=" * 70)
    print("VOICE STRESS ANALYSIS TEST")
    print("=" * 70)

    if os.path.exists(test_file):
        result = analyze_voice(test_file)

        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"  Stress Level: {result['stress_level']}")
        print(f"  Volume Level: {result['volume_level']}")
        print(f"  Speech Pace: {result['speech_pace']}")
        print(f"  Overall Voice Risk: {result['voice_risk']:.0%}")

        print(f"\n🔬 TECHNICAL FEATURES:")
        feats = result["features"]
        print(f"  RMS Energy: {feats['rms']:.4f} (std: {feats['rms_std']:.4f}, max: {feats['rms_max']:.4f})")
        print(f"  Pitch: {feats['pitch']:.1f} Hz (std: {feats['pitch_std']:.1f}, "
              f"range: {feats['pitch_min']:.1f}-{feats['pitch_max']:.1f})")
        print(f"  Speech Rate: {feats['speech_rate']:.2f}")
        print(f"  Pause Ratio: {feats['pause_ratio']:.2f}")
        print(f"  Zero Crossing Rate: {feats['zcr']:.4f}")
        print(f"  Spectral Centroid: {feats['spectral_centroid']:.1f}")

        print("\n" + "=" * 70)
    else:
        print(f"\n❌ Test file '{test_file}' not found.")
        print("Place a test audio file as 'test.wav' to run local tests.")
        print("=" * 70)
