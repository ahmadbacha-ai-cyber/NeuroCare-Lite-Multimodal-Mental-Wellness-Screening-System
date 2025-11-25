# voice_detector.py
"""
Advanced voice stress & energy analysis for NeuroCare Lite.

This module:
- Loads audio safely across platforms
- Extracts multiple vocal biomarkers:
    * RMS loudness (and dB)
    * Pitch (mean + variability)
    * Speech activity ratio (voiced vs silence)
    * Approximate speech rate
    * Zero-crossing rate (noisiness/tension)
- Produces:
    * stress_level (categorical)
    * volume_level (categorical)
    * speech_pace (categorical)
    * voice_risk (0–1)
    * features (rich dict used by the UI)

It is compatible with the current app.py expectations:
    analyze_voice(path) -> {
        "stress_level": str,
        "volume_level": str,
        "speech_pace": str,
        "voice_risk": float,
        "features": {
            "rms": float,
            "pitch": float,
            "speech_rate": float,
            "pause_ratio": float,
            ... (extra advanced features)
        }
    }
"""

import os
import warnings
import logging
from typing import Tuple, Dict, Any

import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

TARGET_SR = 22050


# =============================================================================
#   CROSS-PLATFORM AUDIO LOADER (ROBUST + NORMALIZED)
# =============================================================================
def _safe_load_audio(audio_path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Load audio in a robust, cross-platform way.

    - Uses soundfile where possible (fast and precise)
    - Falls back to librosa.load
    - Normalizes sample rate and trims leading/trailing silence
    """
    y = None
    sr = None

    # Try with soundfile first
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1:  # stereo -> mono
            y = y.mean(axis=1)
        y = y.astype(np.float32)
    except Exception as e:
        logger.info(f"[VOICE] soundfile read failed, falling back to librosa: {e}")

    # Fallback to librosa
    if y is None or sr is None:
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            y = y.astype(np.float32)
        except Exception as e:
            logger.error(f"[VOICE] Failed to load audio file: {e}")
            return np.array([], dtype=np.float32), 0

    # Resample to target sample rate
    if sr != target_sr:
        try:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as e:
            logger.info(f"[VOICE] Resample failed, keeping original sr={sr}: {e}")

    # Trim leading/trailing silence
    try:
        y, _ = librosa.effects.trim(y, top_db=40)
    except Exception as e:
        logger.info(f"[VOICE] Silence trimming failed: {e}")

    if y is None or len(y) < int(0.3 * sr):
        # Too short / invalid
        return np.array([], dtype=np.float32), 0

    return y, sr


# =============================================================================
#   CORE FEATURE EXTRACTORS
# =============================================================================
def _extract_basic_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract low-level acoustic features:
    - RMS energy
    - Zero-crossing rate
    - Spectral centroid (rough brightness)
    - Duration (seconds)
    """
    duration_sec = len(y) / float(sr)

    # RMS energy
    rms_frame = librosa.feature.rms(y=y)[0]
    rms = float(np.mean(rms_frame))
    rms_std = float(np.std(rms_frame))
    rms_max = float(np.max(rms_frame))
    rms_db = 20.0 * np.log10(rms + 1e-9)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    # Spectral centroid (rough brightness estimate)
    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_centroid_mean = float(np.mean(centroid))
    except Exception:
        spec_centroid_mean = 0.0

    return {
        "duration_sec": duration_sec,
        "rms": rms,
        "rms_std": rms_std,
        "rms_max": rms_max,
        "rms_db": rms_db,
        "zcr_mean": zcr_mean,
        "spec_centroid_mean": spec_centroid_mean,
    }


def _extract_pitch(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract pitch curve using librosa.pyin (if possible).

    Returns:
        pitch_mean_hz, pitch_std_hz, voiced_ratio
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
            sr=sr,
        )
    except Exception as e:
        logger.info(f"[VOICE] pYIN pitch extraction failed: {e}")
        return {
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "voiced_ratio": 0.0,
        }

    # Filter out unvoiced frames
    if f0 is None or voiced_flag is None:
        return {
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "voiced_ratio": 0.0,
        }

    f0 = np.array(f0)
    voiced_flag = np.array(voiced_flag).astype(bool)
    voiced_f0 = f0[voiced_flag]

    if len(voiced_f0) == 0:
        return {
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "voiced_ratio": 0.0,
        }

    pitch_mean_hz = float(np.nanmean(voiced_f0))
    pitch_std_hz = float(np.nanstd(voiced_f0))
    voiced_ratio = float(np.mean(voiced_flag.astype(float)))

    return {
        "pitch_mean_hz": pitch_mean_hz,
        "pitch_std_hz": pitch_std_hz,
        "voiced_ratio": voiced_ratio,
    }


def _estimate_speech_rate(y: np.ndarray, sr: int, rms: float) -> Dict[str, float]:
    """
    Approximate speech rate by counting 'speech-active' frames per second.

    This is *not* word count; it's a proxy based on energy bursts:
    - Higher speech_rate -> more syllable-like energy peaks.
    """
    # Frame-level RMS to detect speech segments
    frame_rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]

    # Speech threshold based on global RMS
    thr = max(rms * 0.5, 1e-5)
    speech_mask = frame_rms > thr

    # Count contiguous segments as 'speech bursts'
    speech_bursts = 0
    in_speech = False
    for val in speech_mask:
        if val and not in_speech:
            in_speech = True
            speech_bursts += 1
        elif not val:
            in_speech = False

    duration_sec = len(y) / float(sr)
    if duration_sec <= 0:
        speech_rate = 0.0
    else:
        speech_rate = speech_bursts / duration_sec  # bursts per second

    voiced_ratio = float(np.mean(speech_mask.astype(float)))
    pause_ratio = 1.0 - voiced_ratio

    return {
        "speech_rate": float(speech_rate),
        "pause_ratio": float(pause_ratio),
        "voiced_activity_ratio": voiced_ratio,
    }


# =============================================================================
#   CLASSIFICATION HELPERS
# =============================================================================
def _classify_volume(rms_db: float) -> Tuple[str, float]:
    """
    Classify volume level based on RMS dB.
    Also returns a risk contribution (0–1).
    """
    if rms_db < -40:
        return "Very Quiet", 0.10
    elif rms_db < -30:
        return "Quiet", 0.05
    elif rms_db < -15:
        return "Normal", 0.0
    elif rms_db < -5:
        return "Loud", 0.15
    else:
        return "Very Loud", 0.25


def _classify_speech_pace(speech_rate: float) -> Tuple[str, float]:
    """
    Classify speech pace (slow / normal / fast) using bursts per second.
    Also returns a risk contribution.
    """
    if speech_rate < 0.8:
        return "Very Slow", 0.15
    elif speech_rate < 1.5:
        return "Slow", 0.10
    elif speech_rate < 3.0:
        return "Normal", 0.0
    elif speech_rate < 4.5:
        return "Fast", 0.15
    else:
        return "Very Fast", 0.25


def _analyze_pitch_stress(pitch_mean: float, pitch_std: float, voiced_ratio: float) -> float:
    """
    Compute a stress contribution from pitch characteristics.
    - High pitch_std with moderate/high pitch → arousal/stress
    - Extremely low pitch_mean + low std → flattening (could be depressive)
    """
    if voiced_ratio < 0.3:
        # Not enough voiced data, low confidence
        return 0.05

    # Normalize pitch deviations
    pitch_var = pitch_std / (pitch_mean + 1e-5)

    risk = 0.0
    # High variability suggests tension or emotional arousal
    if pitch_var > 0.25:
        risk += 0.20
    elif pitch_var > 0.15:
        risk += 0.10

    # Very high pitch may indicate arousal
    if pitch_mean > 260:  # Hz, roughly above high-speaking range
        risk += 0.10

    # Very low pitch + very low var might be 'flat' depressive tone
    if pitch_mean < 100 and pitch_var < 0.08:
        risk += 0.10

    return float(min(1.0, max(0.0, risk)))


def _energy_variation_risk(rms_std: float) -> float:
    """
    More dynamic RMS may reflect tension / emphasis.
    Very low variation + low RMS -> flat, low-energy speech.
    """
    if rms_std < 0.005:
        return 0.10  # very flat
    elif rms_std < 0.02:
        return 0.05
    elif rms_std < 0.05:
        return 0.10
    else:
        return 0.15


def _determine_stress_label(total_risk: float, speech_rate: float, pitch_mean: float) -> str:
    """
    Map numeric voice_risk and patterns to a human-readable label.
    Distinguish between 'activated' and 'fatigued' high stress.
    """
    if total_risk >= 0.75:
        if speech_rate >= 3.0 or pitch_mean > 230:
            return "High Stress (Activated)"
        else:
            return "High Stress (Fatigued/Flat)"
    elif total_risk >= 0.50:
        return "Moderate-to-High Stress"
    elif total_risk >= 0.30:
        return "Mild-to-Moderate Stress"
    else:
        return "Low Voice Stress"


# =============================================================================
#   MAIN PUBLIC FUNCTION
# =============================================================================
def analyze_voice(audio_path: str) -> Dict[str, Any]:
    """
    High-level entry point used by the Streamlit app.

    Parameters
    ----------
    audio_path : str
        Path to a recorded audio file (e.g., temp_recorded_audio.wav)

    Returns
    -------
    dict:
        {
            "stress_level": str,
            "volume_level": str,
            "speech_pace": str,
            "voice_risk": float (0–1),
            "features": {
                "rms": float,
                "rms_db": float,
                "pitch": float,
                "pitch_std": float,
                "speech_rate": float,
                "pause_ratio": float,
                "voiced_ratio": float,
                "zcr_mean": float,
                "duration_sec": float,
                "quality_flag": str,
                ...
            }
        }
    """
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"[VOICE] Audio path not found: {audio_path}")
        return {
            "stress_level": "N/A",
            "volume_level": "N/A",
            "speech_pace": "N/A",
            "voice_risk": 0.0,
            "features": {}
        }

    y, sr = _safe_load_audio(audio_path)
    if sr == 0 or y.size == 0:
        logger.error("[VOICE] Unable to load or too-short audio.")
        return {
            "stress_level": "Low Voice Stress (Invalid/Short Audio)",
            "volume_level": "Very Quiet",
            "speech_pace": "N/A",
            "voice_risk": 0.0,
            "features": {
                "duration_sec": 0.0,
                "quality_flag": "invalid_or_too_short",
            },
        }

    # 1) Basic features
    basic = _extract_basic_features(y, sr)

    # 2) Pitch features
    pitch_feats = _extract_pitch(y, sr)

    # 3) Speech activity / rate features
    rate_feats = _estimate_speech_rate(y, sr, basic["rms"])

    # Merge all features
    feats = {**basic, **pitch_feats, **rate_feats}

    # Volume classification (uses RMS dB)
    volume_label, volume_risk = _classify_volume(feats["rms_db"])

    # Speech pace classification
    speech_pace_label, pace_risk = _classify_speech_pace(feats["speech_rate"])

    # Pitch-based stress
    pitch_risk = _analyze_pitch_stress(
        feats["pitch_mean_hz"],
        feats["pitch_std_hz"],
        feats["voiced_activity_ratio"],
    )

    # RMS variation risk
    rms_variation_risk = _energy_variation_risk(feats["rms_std"])

    # Combine all into a single voice_risk
    # You can tune these weights for your clinical profile
    weights = {
        "volume": 0.15,
        "pace": 0.20,
        "pitch": 0.35,
        "energy_var": 0.30,
    }

    voice_risk = (
        weights["volume"] * volume_risk +
        weights["pace"] * pace_risk +
        weights["pitch"] * pitch_risk +
        weights["energy_var"] * rms_variation_risk
    )

    voice_risk = float(max(0.0, min(1.0, voice_risk)))

    # Determine stress label based on combined picture
    stress_label = _determine_stress_label(
        voice_risk,
        feats["speech_rate"],
        feats["pitch_mean_hz"],
    )

    # -----------------------------------------------------------------
    # Build features dict in a way compatible with your current app
    # -----------------------------------------------------------------
    features_out = {
        # Core UI-visible features
        "rms": float(feats["rms"]),
        "pitch": float(feats["pitch_mean_hz"]),
        "speech_rate": float(feats["speech_rate"]),
        "pause_ratio": float(feats["pause_ratio"]),
        # Extra advanced metrics for debugging / research
        "rms_db": float(feats["rms_db"]),
        "rms_std": float(feats["rms_std"]),
        "rms_max": float(feats["rms_max"]),
        "pitch_std": float(feats["pitch_std_hz"]),
        "voiced_ratio": float(feats["voiced_activity_ratio"]),
        "zcr_mean": float(feats["zcr_mean"]),
        "duration_sec": float(feats["duration_sec"]),
        "spec_centroid_mean": float(feats["spec_centroid_mean"]),
        "quality_flag": (
            "ok" if feats["duration_sec"] >= 2.0 and feats["rms_db"] > -45
            else "caution_low_energy_or_short"
        ),
    }

    result = {
        "stress_level": stress_label,
        "volume_level": volume_label,
        "speech_pace": speech_pace_label,
        "voice_risk": voice_risk,
        "features": features_out,
    }

    return result


# Optional: quick CLI test
if __name__ == "__main__":
    test_file = "test.wav"
    if os.path.exists(test_file):
        print(f"Testing analyze_voice() on: {test_file}")
        res = analyze_voice(test_file)
        print("\n=== VOICE ANALYSIS RESULT ===")
        for k in ["stress_level", "volume_level", "speech_pace", "voice_risk"]:
            print(f"{k}: {res[k]}")
        print("\n--- FEATURES ---")
        for k, v in res["features"].items():
            print(f"{k}: {v}")
    else:
        print("Place a test file named 'test.wav' next to voice_detector.py to run this test.")
