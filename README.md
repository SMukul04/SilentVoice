<div align="center">

# 🤟 SilentVoice

### *Giving Every Gesture a Voice.*

**An AI-Powered Real-Time Bidirectional Communication System for Deaf and Hearing Individuals**

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Development-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-FF6F00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-Whisper-000000?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Unity-3D%20Avatar-000000?style=for-the-badge&logo=unity" />
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-roadmap">Roadmap</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-project-structure">Project Structure</a>
</p>

</div>

---

# 📖 Overview

**SilentVoice** is an AI-powered accessibility platform that enables **real-time communication between deaf and hearing individuals**.

The system provides **bidirectional translation** by converting:

- 🤟 **Sign Language → Text → Speech**
- 🎤 **Speech → Text → 3D Sign Language Avatar**

SilentVoice combines **Computer Vision**, **Deep Learning**, **Speech Recognition**, **Natural Language Processing**, and **3D Avatar Animation** to create an inclusive communication experience.

Unlike traditional sign language translators, SilentVoice focuses on **real-time two-way conversations** instead of one-way translation.

# ✅ Current Progress

The complete preprocessing pipeline for real-time sign language recognition has been implemented and tested.

Completed components:

- Camera Module
- MediaPipe Multi-Hand Detection
- Landmark Extraction
- Left & Right Hand Classification
- Landmark Normalization
- 126-Dimensional Feature Representation
- 30-Frame Sliding Buffer
- Unit Testing
- Live Integration Testing

The project is currently entering the **Dataset Collection** phase for training the gesture recognition model.

---

# 🎯 Objectives

- Enable seamless communication between deaf and hearing individuals.
- Recognize sign language in real time.
- Convert speech into text instantly.
- Display sign language using a realistic 3D avatar.
- Create an intuitive live conversation interface.
- Promote accessibility using Artificial Intelligence.

---

# ✨ Features

## 🤟 Sign Language Recognition

### ✅ Completed

- Real-time Camera Pipeline
- OpenCV Integration
- MediaPipe Multi-Hand Tracking
- Left & Right Hand Detection
- 21 Landmark Extraction per Hand
- Handedness Classification
- Landmark Normalization
- Two-Hand Feature Vector (126 Features)
- 30-Frame Sliding Buffer
- Complete Preprocessing Pipeline
- Unit & Live Integration Tests

### 🚧 In Progress

- Dataset Collection
- LSTM Gesture Recognition
- Sentence Builder
- Live Chat Output

---

## 🎤 Speech Recognition

- Real-time microphone input
- Voice Activity Detection
- OpenAI Whisper Speech-to-Text
- Low-latency transcription
- Automatic Chat Updates

---

## 🔊 Text to Speech

- Coqui TTS
- Natural Voice Generation
- Sign-to-Speech Communication

---

## 👤 Text to Sign Language Avatar

- Sentence Parsing
- Word Tokenization
- Sign Dictionary
- Animation Queue
- Unity Animator
- Blender Avatar
- Smooth Transition Between Signs

---

## 💬 Live Chat

- Real-Time Conversation Window
- Bidirectional Communication
- Timestamped Messages
- Conversation History

---

# 🏗 System Architecture

```text
                    SilentVoice

                 Hearing Person
                       │
               Speech / Text Input
                       │
              Whisper Speech Recognition
                       │
                   Chat Engine
                       │
                       ▼
────────────────────────────────────────────────────────────

                    Backend Controller

────────────────────────────────────────────────────────────
                       ▲
                       │
                 Chat Engine
                       │
        Sign Language Recognition (LSTM)
                       │
             MediaPipe + OpenCV
                       │
                 Deaf Person
```

---

# 🧠 AI Pipelines

## Module 1 — Sign → Text

```text
Camera
   │
OpenCV
   │
MediaPipe Hands
   │
21 Hand Landmarks
   │
Feature Engineering
   │
Landmark Normalization
   │
30 Frame Buffer
   │
LSTM
   │
Predicted Sign
   │
Sentence Builder
   │
Chat Window
```

---

## Module 2 — Sentence Builder

```text
Predicted Signs
      │
Duplicate Filter
      │
Confidence Filter
      │
Sentence Builder
      │
Grammar Correction
      │
Chat Window
```

---

## Module 3 — Speech → Text

```text
Microphone
      │
Voice Activity Detection
      │
OpenAI Whisper
      │
Speech-to-Text
      │
Chat Window
```

---

## Module 4 — Text → Sign Avatar

```text
Speech / Text
      │
Sentence Parser
      │
Tokenizer
      │
Sign Dictionary
      │
Animation Manager
      │
Unity Animator
      │
3D Avatar
```

---

# 🛠 Tech Stack

## Programming

- Python
- JavaScript
- C#

---

## Frontend

- HTML
- CSS
- Bootstrap
- JavaScript

---

## Backend

- Flask
- REST API

---

## Computer Vision

- OpenCV
- MediaPipe

---

## Deep Learning

- TensorFlow
- Keras
- LSTM

---

## Speech Processing

- OpenAI Whisper
- Coqui TTS

---

## Avatar

- Unity
- Blender
- Ready Player Me

---

## Database

- SQLite

Future

- MongoDB

---

# 📂 Project Structure

```text
SilentVoice-AI/

├── backend/
│   ├── sign_recognition/
│   ├── speech_recognition/
│   ├── sentence_builder/
│   ├── avatar_controller/
│   ├── models/
│   └── app.py
│
├── frontend/
│
├── unity/
│
├── blender/
│
├── datasets/
│
├── docs/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

# 📸 Screenshots

> Screenshots will be added during development.

| Home | Sign Recognition | Avatar |
|------|------------------|---------|
| Coming Soon | Coming Soon | Coming Soon |

---

# 🎥 Demo

Demo video will be available after the first stable release.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/SilentVoice-AI.git
```

Move inside the project

```bash
cd SilentVoice-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python app.py
```

---

# 🗺 Development Roadmap

## ✅ Phase 1 — Foundation

- [x] Project Planning
- [x] System Architecture
- [x] Repository Setup
- [x] Backend Structure
- [x] Camera Module
- [x] MediaPipe Integration
- [x] Landmark Extraction
- [x] FrameFeatures
- [x] Landmark Normalization
- [x] FrameBuffer
- [x] Unit Testing
- [x] Integration Testing

---

## 🚧 Phase 2 — Dataset & Model

- [ ] Dataset Collection
- [ ] Dataset Loader
- [ ] Data Augmentation
- [ ] Train/Test Split
- [ ] LSTM Training
- [ ] Model Evaluation
- [ ] Real-Time Prediction

---

## ⏳ Phase 3 — NLP

- [ ] Confidence Filtering
- [ ] Duplicate Removal
- [ ] Sentence Builder
- [ ] Grammar Correction

---

## ⏳ Phase 4 — Speech Processing

- [ ] Whisper Integration
- [ ] Speech-to-Text
- [ ] Text-to-Speech
- [ ] Conversation Engine

---

## ⏳ Phase 5 — Avatar

- [ ] Text Parser
- [ ] Sign Dictionary
- [ ] Animation Queue
- [ ] Unity Avatar

---

## ⏳ Phase 6 — Deployment

- [ ] Web Interface
- [ ] Optimization
- [ ] Documentation
- [ ] Public Release
---

# 🌟 Future Enhancements

- Continuous Sign Language Recognition
- Transformer-based Gesture Recognition
- Indian Sign Language Support
- Multi-language Translation
- Facial Expression Recognition
- Emotion Detection
- Offline Mode
- Mobile Application
- Video Call Integration
- Cloud Deployment
- AI Grammar Correction
- Personalized Avatar

---

# 📊 Project Status

```text
Planning             ████████████████████ 100%

Architecture         ████████████████████ 100%

Preprocessing        ████████████████████ 100%

Dataset Pipeline     ░░░░░░░░░░░░░░░░░░░░   0%

Model Training       ░░░░░░░░░░░░░░░░░░░░   0%

Speech Pipeline      ░░░░░░░░░░░░░░░░░░░░   0%

Avatar Pipeline      ░░░░░░░░░░░░░░░░░░░░   0%

Deployment           ░░░░░░░░░░░░░░░░░░░░   0%
```

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve SilentVoice:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Mukul Singh**

AI & Machine Learning Enthusiast

GitHub: https://github.com/SMukul04

LinkedIn: https://www.linkedin.com/in/mukul-singh-11b71030b/

---

<div align="center">

## ⭐ If you found this project useful, consider giving it a Star!

### Together, let's build a more accessible and inclusive world through AI.

**SilentVoice — Giving Every Gesture a Voice.**

</div>
