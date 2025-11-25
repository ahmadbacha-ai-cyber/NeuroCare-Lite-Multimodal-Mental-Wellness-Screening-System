import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
from camera_detector import analyze_face_image
import pandas as pd
import numpy as np
import datetime
import os
import io
import sys
import streamlit as st
st.set_page_config(page_title="NeuroCare Lite: Mental Wellness Screener", layout="wide", initial_sidebar_state="expanded")
import multiprocessing

# ============================================================
#   CROSS-PLATFORM MULTIPROCESSING FIX (Windows + macOS)
# ============================================================
if sys.platform in ("darwin", "win32"):
    multiprocessing.set_start_method("spawn", force=True)

# ============================================================
#  LAZY IMPORTS (MODEL LOADING HAPPENS INSIDE FUNCTIONS ONLY)
# ============================================================
def get_ai_modules():
    """Lazy import of AI modules to avoid Streamlit reload/spawn problems."""
    try:
        from nlp_screener import analyze_text
    except Exception as e:
        st.warning(f"nlp_screener import failed: {e}. Using fallback analyzer.")
        def analyze_text(_text):
            return {
                "overall_sentiment": "Neutral",
                "depression_risk": 0.0,
                "anxiety_risk": 0.0,
                "stress_risk": 0.0,
                "combined_risk": 0.0
            }

    try:
        from voice_detector import analyze_voice
    except Exception:
        def analyze_voice(_path):
            return {
                "stress_level": "N/A",
                "volume_level": "N/A",
                "speech_pace": "N/A",
                "voice_risk": 0.0,
                "features": {}
            }

    return analyze_text, analyze_voice

# ============================================================
#   MAIN APP (MUST BE INSIDE main() FOR SAFETY)
# ============================================================
def main():
    analyze_text, analyze_voice = get_ai_modules()

    # --------------------------------------------------------
    # Streamlit UI Configuration
    # --------------------------------------------------------
    

    # --------------------------------------------------------
    # Session State Initialization
    # --------------------------------------------------------
    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame({
            "Date": [datetime.date.today() - datetime.timedelta(days=i) for i in range(7, 0, -1)],
            "Depression": np.random.uniform(0.0, 0.3, 7),
            "Anxiety": np.random.uniform(0.0, 0.3, 7),
            "Stress": np.random.uniform(0.0, 0.3, 7),
            "Total_Risk": np.random.uniform(0.0, 0.3, 7),
        })
    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False
    if "recorded_audio_path" not in st.session_state:
        st.session_state.recorded_audio_path = None

    # --------------------------------------------------------
    # Risk Score Logic
    # --------------------------------------------------------
    def calculate_total_risk(text_result, voice_result, text_present, voice_present):
        TEXT_WEIGHT = 0.65
        VOICE_WEIGHT = 0.35
        
        text_risk = text_result.get("combined_risk", 0.0) if text_present else 0.0
        voice_risk = voice_result.get("voice_risk", 0.0) if voice_present else 0.0
        
        if text_present and voice_present:
            return min((text_risk * TEXT_WEIGHT) + (voice_risk * VOICE_WEIGHT), 1.0)
        elif text_present:
            return text_risk
        elif voice_present:
            return voice_risk
        return 0.0

    def get_risk_level(score):
        if score >= 0.65:
            return "High", "🔴"
        elif score >= 0.35:
            return "Moderate", "🟡"
        return "Low", "🟢"

    def get_condition_severity(score):
        """Returns severity label for individual conditions"""
        if score >= 0.70:
            return "Severe"
        elif score >= 0.50:
            return "High"
        elif score >= 0.30:
            return "Moderate"
        elif score >= 0.15:
            return "Mild"
        return "Minimal"

    # --------------------------------------------------------
    # Sidebar Input
    # --------------------------------------------------------
    with st.sidebar:
        st.title("🧠 NeuroCare Lite")
        st.markdown("---")
        st.header("📝 Daily Check-in")

        mood_note = st.text_area(
            "1. How are you feeling today?",
            placeholder="Share your thoughts, feelings, or concerns...\n\nExamples:\n• I'm feeling anxious about work\n• I can't sleep and feel hopeless\n• Everything feels overwhelming",
            height=180
        )

        st.markdown("---")

        st.subheader("🎤 Voice Analysis (Optional)")
        st.caption("Speak naturally for 5-15 seconds. We'll analyze your tone, pace, and vocal stress.")
        
        audio_data = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True,
            format="wav"
        )

        if audio_data:
            st.info("🔄 Processing audio recording...")
            temp_mp3_file = "temp_recorded_audio.mp3"
            
            if os.path.exists(temp_mp3_file):
                os.remove(temp_mp3_file)

            try:
                audio_bytes = audio_data['bytes']
                wav_audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
                wav_audio.export(temp_mp3_file, format="mp3")
                st.session_state.recorded_audio_path = temp_mp3_file
                st.success(f"✅ Audio ready for analysis")
            except Exception as e:
                st.error(f"❌ Audio processing error: {e}")
                st.caption("Note: Ensure 'ffmpeg' is installed on your system")
                st.session_state.recorded_audio_path = None

        st.markdown("---")

        if st.button("🔍 Analyze My Wellness", use_container_width=True, type="primary"):
            if not mood_note and not st.session_state.recorded_audio_path:
                st.error("⚠️ Please provide text or audio input to analyze")
            else:
                st.session_state.run_analysis = True

        st.markdown("---")
        st.info("🔒 **Privacy First**: All analysis runs locally. No data leaves your device.")

    # --------------------------------------------------------
    # Main Page
    # --------------------------------------------------------
    st.title("🧠 NeuroCare Lite: Mental Wellness Assessment")
    st.subheader("AI-Powered Depression, Anxiety & Stress Detection")

    if st.session_state.get("run_analysis", False):
        # Initialize results
        text_result = {
            "overall_sentiment": "N/A",
            "depression_risk": 0.0,
            "anxiety_risk": 0.0,
            "stress_risk": 0.0,
            "combined_risk": 0.0
        }
        
        voice_result = {
            "stress_level": "N/A",
            "volume_level": "N/A",
            "speech_pace": "N/A",
            "voice_risk": 0.0,
            "features": {}
        }

        # Text Analysis
        text_present = bool(mood_note)
        if text_present:
            with st.spinner("🔍 Analyzing text for mental health indicators..."):
                try:
                    text_result = analyze_text(mood_note)
                except Exception as e:
                    st.error(f"❌ Text analysis error: {e}")

        # Voice Analysis
        voice_present = False
        recorded_file = st.session_state.recorded_audio_path
        if recorded_file and os.path.exists(recorded_file):
            voice_present = True
            with st.spinner("🎵 Analyzing voice patterns and stress indicators..."):
                try:
                    voice_result = analyze_voice(recorded_file)
                except Exception as e:
                    st.error(f"❌ Voice analysis error: {e}")
                finally:
                    if os.path.exists(recorded_file):
                        os.remove(recorded_file)
                    st.session_state.recorded_audio_path = None

        # Calculate Overall Risk
        total_risk = calculate_total_risk(text_result, voice_result, text_present, voice_present)
        st.session_state.latest_total_risk = total_risk
        risk_level, risk_emoji = get_risk_level(total_risk)

        # --------------------------------------------------------
        # Results Display
        # --------------------------------------------------------
        st.markdown("---")
        st.header(f"📊 Today's Mental Wellness Report {risk_emoji}")

        # Overall Risk Score
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Risk",
                risk_level,
                f"{total_risk:.1%}",
                delta_color="inverse"
            )
        
        with col2:
            depression_severity = get_condition_severity(text_result["depression_risk"])
            st.metric(
                "Depression",
                depression_severity,
                f"{text_result['depression_risk']:.1%}",
                delta_color="inverse"
            )
        
        with col3:
            anxiety_severity = get_condition_severity(text_result["anxiety_risk"])
            st.metric(
                "Anxiety",
                anxiety_severity,
                f"{text_result['anxiety_risk']:.1%}",
                delta_color="inverse"
            )
        
        with col4:
            stress_severity = get_condition_severity(text_result["stress_risk"])
            st.metric(
                "Stress",
                stress_severity,
                f"{text_result['stress_risk']:.1%}",
                delta_color="inverse"
            )

        # Detailed Analysis
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📝 Text Analysis Results")
            if text_present:
                st.write(f"**Overall Sentiment:** {text_result['overall_sentiment']}")
                st.write(f"**Depression Indicators:** {text_result['depression_risk']:.1%}")
                st.write(f"**Anxiety Indicators:** {text_result['anxiety_risk']:.1%}")
                st.write(f"**Stress Indicators:** {text_result['stress_risk']:.1%}")
            else:
                st.info("No text input provided")
        
        with col_right:
            st.subheader("🎤 Voice Analysis Results")
            if voice_present:
                st.write(f"**Stress Level:** {voice_result['stress_level']}")
                st.write(f"**Volume Level:** {voice_result['volume_level']}")
                st.write(f"**Speech Pace:** {voice_result['speech_pace']}")
                st.write(f"**Voice Risk Score:** {voice_result['voice_risk']:.1%}")
                
                if voice_result.get("features"):
                    with st.expander("🔬 Technical Voice Features"):
                        feats = voice_result["features"]
                        st.write(f"• RMS Energy: {feats.get('rms', 0):.4f}")
                        st.write(f"• Pitch (Hz): {feats.get('pitch', 0):.1f}")
                        st.write(f"• Speech Rate: {feats.get('speech_rate', 0):.2f}")
                        st.write(f"• Pause Ratio: {feats.get('pause_ratio', 0):.2f}")
            else:
                st.info("No voice input provided")

        # --------------------------------------------------------
        # Recommendations
        # --------------------------------------------------------
        st.markdown("---")
        st.header("💡 Personalized Recommendations")
        
        if risk_level == "High":
            st.error("🚨 **HIGH RISK DETECTED**")
            st.markdown("""
            **Immediate Actions:**
            - 📞 Contact a mental health professional or counselor
            - 🆘 If in crisis, call emergency services or suicide hotline
            - 👥 Reach out to a trusted friend or family member
            - 🏥 Consider scheduling an appointment with a psychiatrist
            
            **Crisis Resources:**
            - National Suicide Prevention Lifeline: 988
            - Crisis Text Line: Text HOME to 741741
            """)
            
        elif risk_level == "Moderate":
            st.warning("⚠️ **MODERATE RISK - Action Recommended**")
            st.markdown("""
            **Suggested Actions:**
            - 📅 Schedule a check-in with a therapist or counselor
            - 🧘 Practice daily stress-reduction techniques (meditation, deep breathing)
            - 💪 Maintain regular exercise and healthy sleep schedule
            - 📱 Use mental health apps for guided support
            - 👨‍👩‍👧 Stay connected with supportive people
            """)
            
        else:
            st.success("✅ **LOW RISK - Keep Up the Good Work!**")
            st.markdown("""
            **Maintain Your Wellness:**
            - 😊 Continue your current self-care practices
            - 📊 Monitor your mood regularly
            - 🏃 Stay physically active
            - 😴 Prioritize good sleep hygiene
            - 🤝 Nurture your social connections
            """)

        # Specific condition warnings
        if text_result["depression_risk"] >= 0.60:
            st.warning("⚠️ **High depression indicators detected.** Consider speaking with a mental health professional about depression screening.")
        
        if text_result["anxiety_risk"] >= 0.60:
            st.warning("⚠️ **High anxiety indicators detected.** Anxiety management techniques or professional support may be beneficial.")
        
        if text_result["stress_risk"] >= 0.60:
            st.warning("⚠️ **High stress indicators detected.** Consider stress-reduction techniques or counseling.")

        # --------------------------------------------------------
        # Update History
        # --------------------------------------------------------
        today = datetime.date.today()
        if today not in st.session_state.history["Date"].values:
            new_row = pd.DataFrame({
                "Date": [today],
                "Depression": [text_result["depression_risk"]],
                "Anxiety": [text_result["anxiety_risk"]],
                "Stress": [text_result["stress_risk"]],
                "Total_Risk": [total_risk],
            })
            st.session_state.history = pd.concat([st.session_state.history, new_row]).reset_index(drop=True)
        else:
            idx = st.session_state.history[st.session_state.history["Date"] == today].index[0]
            st.session_state.history.loc[idx, ["Depression", "Anxiety", "Stress", "Total_Risk"]] = (
                text_result["depression_risk"],
                text_result["anxiety_risk"],
                text_result["stress_risk"],
                total_risk
            )

        st.session_state.run_analysis = False


# --------------------------------------------------------
# Facial Emotion Scan (Camera)
# --------------------------------------------------------
st.markdown("---")
st.header("📷 Facial Emotion & Expression Scan (Camera)")
st.caption("Optional facial-emotion check using your webcam to refine your wellness profile.")

camera_image = st.camera_input(
    "Align your face in the frame and capture a snapshot",
    key="neurocare_camera_input"
)

if camera_image is not None:
    with st.spinner("🧠 Analyzing facial expressions..."):
        try:
            face_result = analyze_face_image(camera_image.getvalue())

            st.subheader("Facial Analysis Results")
            st.write(f"**Dominant Emotion:** {face_result['dominant_emotion']}")
            st.write(f"**Facial Stress Level:** {face_result['facial_stress_level']}")
            st.write(f"**Facial Risk Score:** {face_result['face_risk']:.1%}")

            if face_result.get("emotion_scores"):
                st.markdown("**Emotion Distribution:**")
                emo_df = pd.DataFrame({
                    "Emotion": list(face_result["emotion_scores"].keys()),
                    "Score": list(face_result["emotion_scores"].values())
                }).set_index("Emotion")
                st.bar_chart(emo_df, height=280, use_container_width=True)

            base_risk = st.session_state.get("latest_total_risk")
            if base_risk is not None:
                combined_with_face = min(1.0, 0.7 * base_risk + 0.3 * face_result["face_risk"])
                combined_level, combined_emoji = get_risk_level(combined_with_face)
                st.info(
                    f"**Updated Overall Risk (Text + Voice + Face):** "
                    f"{combined_emoji} {combined_level} — {combined_with_face:.1%}"
                )
            else:
                st.caption("Run a text/voice analysis first to combine facial results into the overall risk score.")
        except Exception as e:
            st.error(f"❌ Facial analysis error: {e}")
            st.caption("Ensure DeepFace and its dependencies are installed, and that your face is clearly visible.")

    # --------------------------------------------------------
    # Trend Charts
    # --------------------------------------------------------
    st.markdown("---")
    st.header("📈 Wellness Trends (Last 7 Days)")
    
    tab1, tab2 = st.tabs(["📊 Overall Risk", "🔍 Detailed Breakdown"])
    
    with tab1:
        st.line_chart(
            st.session_state.history.set_index("Date")[["Total_Risk"]], 
            height=300,
            use_container_width=True
        )
    
    with tab2:
        st.line_chart(
            st.session_state.history.set_index("Date")[["Depression", "Anxiety", "Stress"]], 
            height=300,
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("""
    ### 📋 Risk Score Guide
    
    | Level | Score Range | Color | Action |
    |-------|-------------|-------|--------|
    | **Minimal** | 0.00 - 0.14 | 🟢 Green | Continue wellness practices |
    | **Mild** | 0.15 - 0.29 | 🟢 Green | Monitor and maintain self-care |
    | **Moderate** | 0.30 - 0.49 | 🟡 Yellow | Consider professional support |
    | **High** | 0.50 - 0.69 | 🟡 Yellow | Seek professional help |
    | **Severe** | 0.70 - 1.00 | 🔴 Red | Immediate professional intervention |
    
    ---
    
    **Disclaimer:** This tool is for screening purposes only and does not replace professional diagnosis. 
    If you're experiencing mental health concerns, please consult a qualified healthcare provider.
    """)

# ============================================================
# REQUIRED SAFE ENTRY POINT (macOS + Windows)
# ============================================================
if __name__ == "__main__":
    main()
