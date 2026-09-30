# 🤟 SilentVoice

### Giving Every Gesture a Voice.

<p align="center">
  <strong>AI-Powered Real-Time Indian Sign Language Recognition</strong><br>
  Turning hand gestures into meaningful text through computer vision and deep learning.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-Completed-success?style=for-the-badge" alt="Project Ongoing">
  <img src="https://img.shields.io/badge/Indian%20Sign%20Language-ISL-blue?style=for-the-badge" alt="ISL">
  <img src="https://img.shields.io/badge/Real--Time-Recognition-orange?style=for-the-badge" alt="Real Time">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT License">
</p>

---

## 🌟 Overview

**SilentVoice** is an AI-powered real-time Indian Sign Language (ISL) recognition web application designed to convert hand gestures into text.

The system uses **MediaPipe for hand landmark extraction**, a **temporal sequence processing pipeline**, and an **LSTM deep learning model** to recognize sign gestures from a live webcam feed.

Recognized signs are stabilized before being added to a sentence, reducing unstable predictions and creating a smoother recognition experience.

SilentVoice combines:

- 👁️ Computer Vision
- 🧠 Deep Learning
- 🌐 Web Development
- ⚡ Real-Time Processing
- 🤖 Temporal Sequence Recognition

> **Goal:** Make communication more accessible by giving visual gestures a digital voice.

---

# ✨ Key Capabilities

## 🎥 Real-Time Sign Recognition

SilentVoice uses the user's webcam to capture hand gestures and recognize supported Indian Sign Language signs in real time.

## 🖐️ Hand Landmark Detection

MediaPipe detects **21 landmarks per hand**, supporting up to **two hands simultaneously**.

These landmarks provide the numerical representation required by the recognition model.

## 🧮 Landmark Normalization

The extracted hand coordinates are normalized relative to the detected hand region.

This produces a more consistent feature representation for model inference.

## 🧠 LSTM-Based Temporal Recognition

SilentVoice analyzes a **32-frame temporal sequence** rather than relying on a single frame.

This allows the LSTM model to capture movement and temporal patterns within a gesture.

## 🛡️ Prediction Stabilization

Raw model predictions are filtered before they are accepted.

The temporal stabilizer helps prevent short-lived or unstable predictions from being immediately added to the sentence.

## 📝 Sentence Builder

Confirmed signs are automatically appended to the current sentence.

This allows multiple recognized gestures to form a continuous text output.

## 📊 Real-Time Metrics

The application provides runtime information including:

- FPS
- Hand detection count
- Prediction status
- Recognition output

## 🎛️ Interactive Controls

The web interface provides:

- ▶️ Start Camera
- ⏹️ Stop Camera
- 🔄 Restart Camera
- 🧹 Clear Sentence

---

# 🔄 Recognition Pipeline

```text
┌──────────────────────┐
│     Webcam Feed      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Browser Video Stream │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ MediaPipe Hand       │
│ Landmarker           │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ 21 Landmarks / Hand  │
│ Up to 2 Hands        │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Landmark Normalizer  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ 126-D Feature Vector │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ 32-Frame Sequence    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ FastAPI /predict     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ LSTM Recognition     │
│ Model                │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Prediction           │
│ Stabilizer           │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Confirmed Sign       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Sentence Builder     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Text Output          │
└──────────────────────┘
```

---

# 🔄 Two-Way Communication Flow

SilentVoice is designed around a **two-way communication architecture**, enabling communication between a sign-language user and a non-sign-language user.

```text
                         🤝 TWO-WAY COMMUNICATION
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼

          🧑 Sign Language User        🧑 Non-Sign Language User
                    │                           │
                    │                           │
              🤟 Hand Signs                🎤 Speech
                    │                           │
                    ▼                           ▼
          ┌─────────────────┐        ┌─────────────────┐
          │    Webcam       │        │ Speech Input    │
          └────────┬────────┘        └────────┬────────┘
                   │                          │
                   ▼                          ▼
          ┌─────────────────┐        ┌─────────────────┐
          │ MediaPipe Hand  │        │ Speech-to-Text  │
          │ Landmark Engine │        │     Engine      │
          └────────┬────────┘        └────────┬────────┘
                   │                          │
                   ▼                          ▼
          ┌─────────────────┐        ┌─────────────────┐
          │ Landmark        │        │ Text Processing │
          │ Processing      │        │ & NLP           │
          └────────┬────────┘        └────────┬────────┘
                   │                          │
                   ▼                          ▼
          ┌─────────────────┐        ┌─────────────────┐
          │ Temporal        │        │ Text / Intent   │
          │ Sequence        │        │ Representation  │
          └────────┬────────┘        └────────┬────────┘
                   │                          │
                   ▼                          ▼
          ┌─────────────────┐        ┌─────────────────┐
          │ LSTM Sign       │        │ Sign Translation│
          │ Recognition     │        │ Pipeline        │
          └────────┬────────┘        └────────┬────────┘
                   │                          │
                   ▼                          ▼
          ┌─────────────────┐        ┌─────────────────┐
          │ Recognized Text │        │ Sign Sequence   │
          └────────┬────────┘        └────────┬────────┘
                   │                          │
                   └────────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   💬 Communication      │
                    │       Interface         │
                    └─────────────────────────┘
```

### Communication Directions

| Direction | Input | Processing | Output |
|---|---|---|---|
| 🤟 Sign → Text | Hand gestures | MediaPipe → LSTM → Stabilizer | Text |
| 🎤 Speech → Sign | Speech | Speech-to-Text → NLP → Sign Translation | Sign representation |

### Current Implementation

The **Sign → Text** pipeline is implemented in SilentVoice:

```text
Webcam
   ↓
MediaPipe
   ↓
Hand Landmarks
   ↓
Normalization
   ↓
32-Frame Sequence
   ↓
LSTM
   ↓
Temporal Stabilization
   ↓
Sentence Builder
   ↓
Text
```

The reverse **Speech/Text → Sign** direction is represented as the communication architecture and extension path; the currently implemented recognition pipeline is **Sign → Text**.

---

# 🧠 How It Works

### 1. Camera Capture

The browser captures the user's live webcam stream.

### 2. Hand Detection

MediaPipe identifies the hands and extracts **21 landmark points per hand**.

### 3. Feature Construction

Coordinates from up to two hands are combined into a **126-dimensional feature vector**.

When only one hand is detected, the missing hand representation is zero-padded.

### 4. Temporal Buffering

The system stores **32 consecutive frames**.

This allows the model to understand the movement of a gesture over time.

### 5. Backend Prediction

The processed sequence is sent to the FastAPI prediction endpoint:

```text
/predict
```

### 6. LSTM Classification

The LSTM model analyzes the temporal sequence and predicts the corresponding sign.

### 7. Temporal Stabilization

The prediction stabilizer filters unstable predictions.

Only consistent predictions are accepted as confirmed signs.

### 8. Sentence Construction

Confirmed signs are appended to the sentence builder.

### 9. Text Output

The recognized sentence is displayed directly in the web interface.

---

# 📚 Supported ISL Signs

The implemented recognition model supports **13 sign classes**:

| # | Sign |
|---:|---|
| 1 | Alive |
| 2 | Clean |
| 3 | Dead |
| 4 | Deep |
| 5 | Dirty |
| 6 | Hard |
| 7 | Heavy |
| 8 | High |
| 9 | Low |
| 10 | Shallow |
| 11 | Soft |
| 12 | Strong |
| 13 | Weak |

---

# 📊 Model Configuration

| Parameter | Value |
|---|---:|
| Sequence Length | 32 Frames |
| Feature Dimension | 126 |
| Hands Supported | Up to 2 |
| Landmarks | 21 per hand |
| Recognition Model | LSTM |
| Recognition Classes | 13 |
| Total Samples | 104 |
| Training Samples | 78 |
| Validation Samples | 13 |
| Test Samples | 13 |
| Test Accuracy | ~92.3% |

---

# 🛠️ Tech Stack

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/Uvicorn-4051B5?style=for-the-badge&logo=uvicorn&logoColor=white" alt="Uvicorn">
<img src="https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge" alt="Jinja2">

<br>

<img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5">
<img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3">
<img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">

<br>

<img src="https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe">
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">

<br>

<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
<img src="https://img.shields.io/badge/LSTM-Deep%20Learning-8A2BE2?style=for-the-badge" alt="LSTM">
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">

</p>

## Technology Breakdown

| Layer | Technologies |
|---|---|
| Frontend | HTML, CSS, Vanilla JavaScript |
| Backend | Python, FastAPI, Uvicorn, Jinja2 |
| Computer Vision | MediaPipe, OpenCV |
| Machine Learning | TensorFlow, Keras, LSTM |
| Numerical Processing | NumPy |
| Testing | Pytest |
| Communication | REST API |

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │        User         │
                    │   Performs Sign     │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │       Webcam        │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ MediaPipe Landmarker│
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Landmark Processing │
                    │ & Normalization     │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ 32-Frame Sequence   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ FastAPI Prediction  │
                    │      Endpoint       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │     LSTM Model      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Temporal Stabilizer │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │  Sentence Builder   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │  Recognized Text    │
                    └─────────────────────┘
```

---

# 📁 Project Structure

```text
SilentVoice/
│
├── backend/
│   └── app/
│       └── main.py
│
├── frontend/
│   ├── HTML templates
│   ├── CSS
│   └── JavaScript
│
├── models/
│   ├── LSTM model architecture
│   └── Saved model weights
│
├── datasets/
│   └── Recognition dataset
│
├── tests/
│   ├── Backend tests
│   ├── Recognition tests
│   └── Frontend integration tests
│
├── artifacts/
│   └── Generated artifacts and logs
│
├── README.md
└── requirements.txt
```

---

# 🧪 Testing

SilentVoice includes automated tests covering the core recognition and application pipeline.

### Test Coverage

- ✅ MediaPipe hand detection
- ✅ Landmark extraction
- ✅ Landmark normalization
- ✅ Frontend landmark extraction
- ✅ Prediction integration
- ✅ Temporal prediction stabilizer
- ✅ Sentence builder
- ✅ Webcam controls
- ✅ FastAPI prediction endpoint
- ✅ Backend integration
- ✅ Input validation
- ✅ Error handling

### Run Tests

```bash
python -m pytest tests backend
```

---

# 🚀 Run Locally

## 1. Clone the Repository

```bash
git clone https://github.com/SMukul04/SilentVoice.git
cd SilentVoice
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

## 3. Activate the Environment

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Start the Application

```bash
python -m uvicorn backend.app.main:app --reload
```

## 6. Open SilentVoice

Open:

```text
http://127.0.0.1:8000/app
```

Allow webcam access and start performing supported signs.

---

# 💡 What Makes SilentVoice Stand Out?

### 🎯 Gesture → Meaning

SilentVoice transforms physical hand movements into numerical features and finally into meaningful text.

### ⏱️ Temporal Understanding

The LSTM analyzes sequences of frames, allowing the system to recognize temporal movement patterns rather than depending only on static images.

### 🛡️ Stable Recognition

Temporal stabilization prevents every raw model prediction from immediately becoming part of the sentence.

### 🌐 Web-Based Experience

Users can interact with the recognition system directly through a browser using a webcam.

### 🔌 Modular Architecture

The system separates:

- Computer vision
- Feature processing
- Model inference
- Backend API
- Prediction stabilization
- Sentence generation
- Frontend interaction

This makes the system easier to maintain and extend.

---

# 📸 Application Preview

Add screenshots of your actual application here.

Recommended structure:

```text
docs/
└── screenshots/
    ├── home.png
    ├── recognition.png
    └── result.png
```

Then add:

```markdown
## 🖥️ Interface

![SilentVoice Interface](docs/screenshots/home.png)

![Real-Time Recognition](docs/screenshots/recognition.png)

![Recognition Result](docs/screenshots/result.png)
```

---

# 🎯 Project Highlights

| Area | Implementation |
|---|---|
| 🤖 AI | LSTM-based temporal classification |
| 👁️ Computer Vision | MediaPipe hand landmarks |
| 🎥 Real-Time Input | Browser webcam |
| 🧮 Feature Engineering | 126-dimensional normalized representation |
| ⏱️ Temporal Processing | 32-frame sequences |
| ⚡ Backend | FastAPI REST API |
| 🌐 Frontend | HTML + CSS + Vanilla JavaScript |
| 📝 Output | Stabilized sign-to-text sentence |
| 🧪 Testing | Automated backend and integration tests |
| 🤟 Recognition | 13 ISL sign classes |

---

# 🌍 Vision

SilentVoice is built around a simple idea:

> **Communication should not be limited by the way we speak.**

By combining computer vision, machine learning, and an accessible web interface, SilentVoice demonstrates how AI can be used to interpret visual communication in real time.

The implemented system provides real-time recognition for **13 ISL sign classes**, converting recognized gestures into text through a complete webcam-to-AI-to-text pipeline.

---

# 👨‍💻 Author

**Mukul Singh**

GitHub: [@SMukul04](https://github.com/SMukul04)

---

# 📄 License

This project is licensed under the **MIT License**.

---

<p align="center">
  <strong>🤟 SilentVoice</strong><br>
  <em>Giving Every Gesture a Voice.</em>
</p>
