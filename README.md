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

* **Surim Bong (Team Leader)**

  * Project planning and progress management
  * Final document and PPT writing
  * VAE model building and scoring algorithm design
  * Model training, backend API development
  * Web integration and overall testing
  * Demonstration video production

* **Gunhoon Song**

  * Speech preprocessing and feature extraction (MFCC, formants, spectrogram)
  * Assist in VAE model building
  * Database design and score storage
  * PPT writing, backend support

* **Yehyung Ho**

  * Korean speech transcription module development
  * Frontend page design and function implementation
  * Visualization of scores and suggestions
  * Web integration, project publishing, presentation script writing

* **Gunbang Jung**

  * Scoring suggestion and improvement module development
  * Frontend optimization and visualization module development
  * Web integration, overall testing, support for midterm PPT and model training

---

## 🔑 Notes

* Recommended audio format: 16kHz, mono-channel WAV.
* Store your OpenAI API KEY securely (e.g., in a `.env` file).
* Prepare reference audios in the `data/kss/1/` directory.

---

## 📧 Contact

For inquiries, please contact the project lead at [suilinxpeng15@gmail.com](mailto:suilinxpeng15@gmail.com).

