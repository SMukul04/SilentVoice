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

- Real-time camera capture
- OpenCV integration
- MediaPipe Hand Tracking
- 21 Hand Landmark Detection
- Landmark Normalization
- LSTM-based Gesture Recognition
- AI Sentence Builder
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

## Phase 1

- [x] Project Planning
- [x] Architecture Design
- [ ] Repository Setup
- [ ] Backend Setup

---

## Phase 2

- [ ] MediaPipe Integration
- [ ] OpenCV Camera
- [ ] Dataset Preparation
- [ ] LSTM Training

---

## Phase 3

- [ ] Whisper Integration
- [ ] Speech Recognition
- [ ] Live Chat Window

---

## Phase 4

- [ ] Blender Avatar
- [ ] Unity Animator
- [ ] Animation Queue
- [ ] Text-to-Sign Module

---

## Phase 5

- [ ] Deployment
- [ ] Performance Optimization
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

Development          ███░░░░░░░░░░░░░░░░░ 15%

Testing              ░░░░░░░░░░░░░░░░░░░░ 0%

Deployment           ░░░░░░░░░░░░░░░░░░░░ 0%
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

GitHub: https://github.com/<YOUR_USERNAME>

LinkedIn: *(Add your LinkedIn here)*

---

<div align="center">

## ⭐ If you found this project useful, consider giving it a Star!

### Together, let's build a more accessible and inclusive world through AI.

**SilentVoice — Giving Every Gesture a Voice.**

</div>
