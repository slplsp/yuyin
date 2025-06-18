# Pronunciation Evaluation System

## 📌 Project Overview

This project is a deep learning-based Korean pronunciation evaluation system that supports audio upload, pronunciation scoring, DTW-based alignment visualization, and AI-based pronunciation improvement suggestions. The speech dataset is from Kaggle: [Korean Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset).

---

## 📂 File Structure

```
.
├── app.py              # Web service main (Flask)
├── train.py            # Model training script
├── inference.py        # Pronunciation evaluation and visualization
├── inference1.py       # Pronunciation evaluation with AI feedback
├── test.py             # Batch TTS generation using gTTS
├── test1.py            # OpenAI GPT API test
├── test2.py            # Audio format conversion (m4a → wav)
├── TTS1.py             # TTS generation using XTTS
├── dataset.py          # Triplet dataset class
├── model.py            # Voice encoder model definition
├── wav2mel.py          # Mel spectrogram extraction module
├── static/             # Frontend files (HTML, CSS, JS)
├── uploads/            # Temporary upload directory
├── output_audio/       # TTS output directory
├── audio_history/      # Audio history for evaluations
├── cache/              # Temporary cache directory
└── history.json        # Evaluation history file
```

---

## ⚙️ Installation

```bash
pip install torch librosa gTTS moviepy flask tenacity fastdtw matplotlib scikit-learn TTS openai tqdm soundfile
```

---

## 🚀 How to Run

### Start Web Service

```bash
python app.py
```

Open browser: [http://127.0.0.1:5001/](http://127.0.0.1:5001/)

### Train Model

```bash
python train.py
```

Outputs:

* `voice_encoder_best.pth` (best model)
* `Cosine Similarity.png`, `t-SNE.png` (visualizations)

### Command-line Pronunciation Evaluation

```bash
python inference.py
```

or

```bash
python inference1.py
```

### Generate TTS

```bash
python TTS1.py
python test.py
```

### Convert Audio Format

```bash
python test2.py
```

---

## 👥 Team Members & Roles

수림봉

* Project planning and progress management
* Report and PPT writing
* VAE model building and scoring algorithm design
* Model training
* Backend interface development
* Web and model integration
* Overall testing and connection of web end
* Production of the demonstration video

송건훈

* Speech preprocessing and feature extraction (MFCC, Formants, Spectrogram)
* Assist in building VAE model
* Provide assistance in backend interface development
* PPT writing
* Database design and score storage
* Web and model integration
* Cooperate with backend interface development

호예형

* Korean speech transcription module development
* Provide assistance in Web and model integration
* Frontend page design and function implementation
* Visualization display of scoring results and suggestions
* Assist with web and model integration
* Project published
* presentation script
* Provided support in producing the demonstration video
* Report writing

정건방

* Scoring suggestion output and improvement module development
* Frontend page optimization
* develop visualization module
* Assist with web and model integration
* Overall testing
* Provided support in producing the midterm PPT and Model training
* Web translation

---

## 🔑 Notes

* Recommended audio format: 16kHz, mono-channel WAV.
* Store your OpenAI API KEY securely (e.g., in a `.env` file).
* Prepare reference audios in the `data/kss/1/` directory.

---

## 📧 Contact

For inquiries, please contact the project lead at [suilinxpeng15@gmail.com](mailto:suilinpeng15@gmail.com).

