# SilentVoice
"Giving Every Gesture a Voice."

## Project Status
**Current Phase:** Phase 3 — Frontend Recognition Experience (Completed)
**Current Status:** Early Development Checkpoint. The project currently features an AI-powered real-time Indian Sign Language (ISL) recognition system. The implemented model supports a 13-class development subset and is not yet a final production model.

## Overview
SilentVoice is an AI-Powered Real-Time Indian Sign Language Recognition and Future Bidirectional Communication System. Its goal is to bridge the communication gap by translating real-time sign language into text and, eventually, providing full bidirectional communication features.

## Current Capabilities
- **Real-Time Sign Recognition:** Detects and classifies 13 development ISL signs in real-time.
- **Webcam Interface:** Accessible directly through a modern web browser.
- **Browser-Side MediaPipe Integration:** Extracts hand landmarks (21 landmarks per hand, up to 2 hands) locally in the browser to reduce latency.
- **Temporal Sequence Recognition:** Uses a 32-frame buffer to capture the temporal dynamics of signs.
- **Temporal Stabilization:** Filters out unstable predictions to ensure accurate sentence building.
- **Sentence Builder:** Combines confirmed signs into a complete sentence string.
- **Frontend Controls:** Start, stop, and restart camera controls with a clear sentence function.
- **Real-Time Metrics:** Displays real-time FPS, hand detection count, and prediction status.

## Current Recognition Pipeline
1. **Camera Capture:** The user's webcam feed is captured.
2. **Browser Video Stream:** Displayed in the HTML frontend.
3. **MediaPipe Hand Landmarker:** Extracts 21 landmarks per hand.
4. **Landmark Normalization:** Normalizes coordinates relative to the hand's bounding box.
5. **126-Dimensional Feature Vector:** Combines coordinates for two hands (missing hands are zero-padded).
6. **32-Frame Temporal Sequence:** Buffers 32 frames to capture movement.
7. **FastAPI `/predict`:** Sends the sequence to the backend prediction REST API.
8. **LSTM Recognition Model:** Evaluates the temporal sequence.
9. **Raw Prediction:** The model returns a raw sign classification.
10. **Temporal Stabilizer:** Filters out unstable predictions by requiring consecutive matches.
11. **Stable / Confirmed Sign:** Only verified signs are accepted.
12. **Sentence Builder:** Appends confirmed signs to the ongoing text.
13. **Sentence Output:** Text is displayed to the user.

*Note: Raw predictions are not immediately added to the sentence. The temporal stabilization reduces unstable predictions, and only new confirmed predictions are appended to the sentence. Removing hands acts as a recognition boundary/reset.*

## Architecture
```
Camera
  ↓
Browser Video Stream
  ↓
MediaPipe Hand Landmarker
  ↓
21 landmarks × up to 2 hands
  ↓
Landmark Normalization
  ↓
126-Dimensional Feature Vector
  ↓
32-Frame Temporal Sequence
  ↓
FastAPI /predict
  ↓
LSTM Recognition Model
  ↓
Raw Prediction
  ↓
Temporal Stabilizer
  ↓
Stable / Confirmed Sign
  ↓
Sentence Builder
  ↓
Sentence Output
```

## Current Model/Dataset Checkpoint
The currently implemented model is a **DEVELOPMENT CHECKPOINT** only.
- **Sequence Length:** 32 frames
- **Feature Dimension:** 126
- **Current Classes:** 13 (alive, clean, dead, deep, dirty, hard, heavy, high, low, shallow, soft, strong, weak)
- **Dataset Size:** 104 total samples (78 training, 13 validation, 13 test)
- **Test Accuracy:** Approximately 92.3% on the 13-sample test set. *This is an early development evaluation and does NOT represent a production-quality/final accuracy claim.*

## Tech Stack
**Backend:**
- Python 3.11+
- FastAPI
- Uvicorn
- Jinja2

**Frontend:**
- HTML
- CSS
- Vanilla JavaScript

**Computer Vision:**
- MediaPipe
- OpenCV

**Machine Learning:**
- TensorFlow/Keras
- LSTM
- NumPy

## Project Structure
```
SilentVoice/
├── backend/            # FastAPI application and routing
├── frontend/           # Vanilla JS, CSS, and HTML templates
├── models/             # LSTM model architecture and saved weights
├── datasets/           # Development dataset pipeline and samples
├── tests/              # Automated unit and integration tests
├── artifacts/          # Generated artifacts and logs
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SMukul04/SilentVoice.git
   cd SilentVoice
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment (Windows PowerShell):**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```bash
   python -m uvicorn backend.app.main:app --reload
   ```

6. **Access the web interface:**
   Navigate to [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app) in your browser.

## Testing
The project includes a robust suite of automated tests covering multiple areas, including:
- MediaPipe detector and landmark extraction
- Frontend landmark extraction and prediction integration
- Prediction stabilizer and sentence builder
- Webcam frontend and controls
- Prediction API and backend integration
- Validation and error handling

To run the tests:
```bash
python -m pytest tests backend
```

## Development Roadmap

**Phase 1 — Foundation**
- [x] Repository setup
- [x] Backend foundation
- [x] Camera/preprocessing pipeline
- [x] MediaPipe hand tracking
- [x] Landmark extraction
- [x] Landmark normalization
- [x] 126-dimensional feature representation
- [x] Temporal buffering
- [x] Dataset pipeline foundation
- [x] Automated testing

**Phase 2 — Recognition**
- [x] Dataset loader
- [x] Development dataset
- [x] LSTM model
- [x] Model evaluation
- [x] Backend prediction API
- [x] Real-time prediction

**Phase 3 — Frontend Recognition Experience**
- [x] Frontend setup
- [x] Webcam interface
- [x] Browser MediaPipe extraction
- [x] Prediction integration
- [x] Temporal stabilization
- [x] Sign confirmation
- [x] Sentence builder
- [x] Start/Stop/Restart controls
- [x] Final frontend verification

**Phase 4 — Recognition Quality Upgrade**
- [ ] Expand to 50-class ISL dataset
- [ ] Video-based dataset pipeline
- [ ] Improve temporal sequence generation
- [ ] Data augmentation
- [ ] Class balancing
- [ ] Model tuning
- [ ] Robust evaluation
- [ ] Unknown/rejection handling
- [ ] Production-quality accuracy evaluation

**Phase 5 — Bidirectional Communication**
- [ ] Speech-to-text
- [ ] Text-to-speech
- [ ] Conversation interface
- [ ] Text-to-sign translation

**Phase 6 — 3D Avatar**
- [ ] Sign dictionary
- [ ] Sentence parser
- [ ] Animation system
- [ ] Unity/3D avatar integration

**Phase 7 — Deployment**
- [ ] Performance optimization
- [ ] Production deployment
- [ ] Documentation
- [ ] Release

## Future Enhancements
The following features are planned for future development but are not yet implemented:
- Production 50-class video-trained Transformer/LSTM model
- Whisper speech recognition (Speech-to-text)
- Coqui TTS (Text-to-speech)
- Unity/Blender/Ready Player Me 3D Avatar integration
- Live bidirectional chat and grammar correction
- Mobile application and Cloud deployment
- Video calling

## Limitations / Current Development Status
- **Limited Classes:** The current recognition model supports only 13 development classes.
- **Small Dataset:** The current evaluation dataset is small, serving as an early proof-of-concept.
- **Accuracy Claims:** The reported test accuracy is an early development checkpoint and should not be interpreted as production accuracy.
- **Future Targets:** The final target is video-based ISL recognition with a substantially larger dataset/class set.
- **Pending Features:** Speech and 3D avatar pipelines are planned but not yet implemented.

## Contributing
Contributions are welcome. Please ensure that tests are run locally and that pull requests align with the development roadmap before submission.

## License
[MIT License](LICENSE)

## Author
[SMukul04](https://github.com/SMukul04)
