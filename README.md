# 🎨 Reddit Artist Sentiment Analysis

A **Deep Learning system** designed to classify Reddit posts about artists into three sentiment categories: Negative, Positive, and Neutral. The project compares the performance of two distinct neural network architectures for sentiment classification.

---

## 🌟 Features

- **Sentiment Classification:** Three classes — Negative, Positive, Neutral.
- **Model Benchmarking:** Compares a custom Bi-LSTM to a Universal Sentence Encoder (USE) transfer learning approach.
- **Robust Pipeline:** Modular scripts for data loading, visualization, training, and evaluation.
- **Optimized Performance:** Utilizes efficient batching and TensorFlow's `AUTOTUNE` for faster training.

---

## 🗂️ Project Structure

```
REDDITSENTIMENT/
├── dataset/
│   ├── data.py                 # Data loading and preprocessing
│   └── explore_data.py         # Exploratory data analysis
├── Functions/
│   └── helperFunctions.py      # Plotting and metrics utilities
├── model/
│   ├── extra_layers.py         # Custom layer definitions
│   └── USE_model.py            # Universal Sentence Encoder (USE) wrapper
├── models/
│   ├── saved_models/           # Saved models
│   ├── model1.py               # Bi-LSTM training script
│   └── model2.py               # USE training script
├── reddit_artist_posts_sentiment.csv  # Raw dataset
└── README.md                   # Project documentation
```

---

## 🛠️ Built With

- **Python 3.x**
- **TensorFlow & Keras**
- **TensorFlow Hub**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **scikit-learn**

---

## 🚀 Getting Started

### Prerequisites

- Python 3.6+
- `pip` package manager

### Installation

```bash
git clone https://github.com/your-username/reddit-sentiment-analysis.git
cd REDDITSENTIMENT

pip install pandas numpy tensorflow keras tensorflow-hub matplotlib scikit-learn
```

---

## 💻 Usage

### 1️⃣ Data Preparation

- Dataset: `reddit_artist_posts_sentiment.csv`
- Loading: `dataset/data.py`
- Visualization: `dataset/explore_data.py`

### 2️⃣ Training

- Change to the `models` directory:
  - **Train Bi-LSTM Model:**\
    `python models/model1.py`
  - **Train Universal Sentence Encoder Model:**\
    `python models/model2.py`

> Scripts automatically handle UTF-8 encoding and TensorFlow logging.

### 3️⃣ Evaluation

- Output includes:
  - Accuracy Score
  - Loss Value
  - Loss Curve Plot (via `Functions/helperFunctions.py`)

---

## 🧠 Models Compared

| Model                   | Key Features                                                              |
|-------------------------|---------------------------------------------------------------------------|
| **Bidirectional LSTM**  | Embedding layer, Bi-LSTM, Global Max Pooling, TextVectorization           |
| **USE (Transfer Learning)** | Universal Sentence Encoder from TF Hub, custom Keras layer, Dense layers |

---
