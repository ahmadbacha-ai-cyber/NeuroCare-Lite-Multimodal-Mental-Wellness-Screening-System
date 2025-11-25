NeuroCare Lite – Multimodal Mental Wellness Screening System
Complete Project Documentation
🎯 Project Overview

NeuroCare Lite is an AI-powered mental wellness screening system that analyzes text, voice, and facial expressions to detect indicators of depression, anxiety, and stress. The system provides privacy-first, fully local analysis, using multimodal machine learning models for early detection and daily wellness monitoring.

Key Objectives

Early detection of mental health concerns

Privacy-preserving, on-device analysis (no cloud dependency)

Multi-modal assessment (text + voice + face)

Real-time risk scoring and trend tracking

Clinically inspired scoring aligned with PHQ-9, GAD-7 & vocal-biomarker research

Target Users

Individuals monitoring daily wellness

Mental health professionals for preliminary screening

AI/ML researchers studying mental-health patterns

Healthcare organizations deploying wellness check-ins

✨ Features
1. Text Analysis (NLP-Based)

Depression Detection: Identifies hopelessness, suicidality, anhedonia, emotional collapse

Anxiety Detection: Detects panic terms, racing thoughts, dread, hypervigilance

Stress Detection: Identifies burnout, overload, emotional fatigue

Hybrid AI:

RoBERTa Transformer (CardiffNLP)

Clinical keyword dictionaries

Suicidality Override: High-risk phrases immediately raise depression risk

Sentiment Analysis: Contextual emotional tone scoring

2. Voice Analysis (Audio Processing)

Speech Pace Detection: Very Slow → Very Fast

Volume Level Detection: Very Quiet → Very Loud

Pitch Analysis (pYIN): Vocal tension, flatness, emotional arousal

Stress Biomarkers:

RMS variability

Pause ratio

Zero-crossing rate

Speech-rate bursts

Voice Risk Scoring: Combined weighted features → stress classification

Outputs:

stress_level

volume_level

speech_pace

voice_risk

10+ detailed acoustic features

3. Facial Emotion Analysis (Camera-Based)

Powered by DeepFace emotion model

Detects: Happy, Sad, Fear, Anger, Disgust, Neutral, Surprise

Computes:

Dominant Emotion

Facial Stress Level

Face Risk Score (0–1)

Weighted scoring:

Negative emotions ↑ risk

Positive emotions ↓ risk

4. Combined Multimodal Risk Scoring

Weighted text-voice fusion:

total_risk = 0.65 * text_risk + 0.35 * voice_risk


Optional facial fusion:

updated_risk = 0.7 * total_risk + 0.3 * face_risk


Severity Levels: Minimal, Mild, Moderate, High, Severe

Personalized Recommendations:

Crisis guidance

Coping strategies

Wellness maintenance

5. Historical Trends & Dashboard

7-day trend tracking

Depression, Anxiety, Stress, Total Risk charts

Professional dashboard UI (Streamlit)

Expandable technical panels for advanced users

🏗️ System Architecture
┌─────────────────────────────────────────────────────┐
│                   User Interface                     │
│                (Streamlit Web App)                   │
└─────────────────┬───────────────────────────────────┘
                  │
      ┌───────────┼──────────────────────────┬───────────┐
      ▼           ▼                          ▼           ▼
┌──────────┐ ┌────────────┐          ┌────────────┐ ┌────────────┐
│ Text     │ │ Voice Input│          │ Camera      │ │ History    │
│ Module   │ │ Recorder   │          │ Snapshot    │ │ Tracker    │
└────┬─────┘ └──────┬─────┘          └──────┬─────┘ └────┬───────┘
     │              │                        │            │
     ▼              ▼                        ▼            ▼
┌────────────┐ ┌─────────────┐      ┌───────────────┐ ┌────────────┐
│ NLP Model  │ │ Voice Model │      │ Facial Model   │ │ Trend Engine│
│ RoBERTa +  │ │ librosa+pYIN│      │ DeepFace       │ │ 7-Day Charts│
│ Keywords   │ │ Features    │      │ Emotion Scores │ │            │
└────┬───────┘ └──────┬──────┘      └───────────────┘ └────┬────────┘
     │                │                                     │
     └──────────┬─────┴──────────────────┬──────────────────┘
                ▼                        ▼
       ┌───────────────────┐     ┌────────────────────┐
       │ Multimodal Fusion │     │ Recommendations     │
       │ Risk Scoring      │     │ Crisis & Wellness   │
       └───────────────────┘     └────────────────────┘

🔧 Technical Documentation
Module 1: app.py (Main Application)

Streamlit orchestration for UI + audio + camera

Lazy loading of AI modules

Produces daily wellness report with metrics

7-day historical trend DataFrame

Functions:

calculate_total_risk()

get_risk_level()

get_condition_severity()

Facial fusion

Module 2: nlp_screener.py (Text Analysis)

RoBERTa-based sentiment classifier

Large clinical keyword dictionaries

Suicidality detection

First-person emotional intensity boost

Hybrid scoring (70% keyword, 30% transformer)

Returns:

depression/anxiety/stress risks

suicidal flag

combined risk

overall sentiment

Module 3: voice_detector.py (Voice Analysis)

Audio loader (cross-platform)

RMS, pitch, speech rate, pause ratio, ZCR, spectral centroid

Vocal biomarker scoring

Speech pace and volume classification

Weighted stress calculation

Returns:

voice_risk

stress_level

detailed 10+ acoustic features

Module 4: camera_detector.py (Facial Analysis)

DeepFace emotion classifier

Emotion distribution

Facial stress label

Facial risk calculation

Integrated with final risk fusion

📖 Usage Instructions
Starting the Application
streamlit run app.py


Then open:

http://localhost:8501

👣 Basic Workflow
1. Text Input

Type your thoughts (free-form journal style)

2. Voice Recording (Optional)

Speak naturally for 5–15 seconds

Wait for audio processing confirmation

3. Camera Snapshot (Optional)

Capture a facial expression for emotion analysis

4. Run Analysis

Click Analyze My Wellness

View depression, anxiety, stress and total risk

5. View Recommendations

Crisis resources

Stress-management advice

Wellness maintenance

6. Track Trends

Scroll to Wellness Trends

Review 7-day chart progression

📊 Interpreting Risk Scores
Score Range	Meaning
0–14%	Minimal
15–29%	Mild
30–49%	Moderate
50–69%	High
70–100%	Severe

Voice + facial stress integrates to refine total risk.

🔬 AI Models & Algorithms
RoBERTa Transformer

125M parameters

Trained on large Twitter corpora

Accurate emotional context detection

Clinical Keyword Engine

Depression: 50+ phrases

Anxiety: 40+ patterns

Stress: 35+ indicators

Positive buffers reduce false positives

Voice Biomarkers

Pitch variation

RMS loudness

Speech-rate dynamics

Pause ratio

ZCR tension

Facial Emotion Analysis

DeepFace CNN emotion model

Weighted negative-emotion scoring

Risk Fusion Formula
total_risk = 0.65 * text_risk + 0.35 * voice_risk
updated_risk = 0.7 * total_risk + 0.3 * face_risk   # optional

👥 Credits

Streamlit

HuggingFace Transformers

librosa

soundfile

DeepFace

PyTorch

Clinical psychology literature (PHQ-9, GAD-7)
