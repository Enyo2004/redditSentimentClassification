# redditSentimentClassification
A Deep Learning NLP project using TensorFlow and Keras to classify Reddit sentiment. It compares a custom Bidirectional LSTM network against Transfer Learning with the Universal Sentence Encoder (USE) to predict Positive, Negative, or Neutral labels, featuring optimized data pipelines.


🎨 Reddit Artist Sentiment Analysis

A Deep Learning project that classifies the sentiment of Reddit posts about artists into three categories: Negative, Positive, or Neutral.

This repository explores Natural Language Processing (NLP) techniques using TensorFlow and Keras, comparing a custom Bidirectional LSTM model against a Transfer Learning approach using the Universal Sentence Encoder (USE).

📂 Directory Structure

Based on the project architecture:

REDDITSENTIMENT/
├── dataset/
│   ├── data.py              # Data loading scripts
│   └── explore_data.py      # EDA and visualization
├── Functions/
│   └── helperFunctions.py   # Utilities (e.g., plotting loss curves)
├── model/
│   ├── extra_layers.py      # Custom Keras layers
│   └── USE_model.py         # Wrapper for Universal Sentence Encoder
├── models/
│   ├── saved_models/        # Binary files for Model 1 and Model 2
│   ├── model1.py            # Training script for Bi-LSTM
│   └── model2.py            # Training script for USE Model
└── reddit_artist_posts_sentiment.csv  # The dataset


📊 The Data

The dataset (reddit_artist_posts_sentiment.csv) consists of text posts scraped from Reddit.

Input (X): Text content of the post.

Label (y): Sentiment category.

Label Mapping:
| Class Name | Numeric Label |
| :--- | :--- |
| Negative | 0 |
| Positive | 1 |
| Neutral | 2 |

🧠 Model Architectures

This project implements and compares two distinct Deep Learning models:

Model 1: Custom Bidirectional LSTM

A sequence model built from scratch.

Text Vectorization: Adapts to the training data vocabulary (capped at 1000 tokens).

Embedding Layer: Converts tokens into dense vectors of fixed size (256).

Bidirectional LSTM: Captures context from both past and future words in the sequence.

Global Max Pooling: Extracts the most significant features from the sequence.

Dense Output: Softmax activation for multiclass classification.

Model 2: Transfer Learning (USE)

Leverages a pre-trained model from TensorFlow Hub.

Universal Sentence Encoder (USE): Encodes text into high-dimensional vectors, capturing deep semantic meaning trained on massive web corpuses.

Custom Dense Layers: Two fully connected layers (128 units, ReLU activation) to adapt the embeddings to this specific dataset.

Dense Output: Softmax activation.

🛠️ Technologies & Libraries

Core: Python 3, TensorFlow 2.x, Keras

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib

Preprocessing: Scikit-Learn (Train/Test Split)

Transfer Learning: TensorFlow Hub

🚀 Installation & Usage

Clone the repository:

git clone [https://github.com/your-username/reddit-sentiment-analysis.git](https://github.com/your-username/reddit-sentiment-analysis.git)
cd REDDITSENTIMENT


Install dependencies:

pip install pandas numpy tensorflow tensorflow-hub matplotlib scikit-learn


Environment Setup:
The scripts automatically configure the following environment variables to handle encoding and logs:

TF_CPP_MIN_LOG_LEVEL = '2'

PYTHONUTF8 = '1'

Training:
You can run the model scripts to train and evaluate:

# Run the LSTM model
python models/model1.py

# Run the USE model
python models/model2.py


📈 Training Features

Performance Optimization: Uses tf.data.Dataset with .batch(128) and .prefetch(tf.data.AUTOTUNE) for efficient data loading.

Callbacks:

EarlyStopping: Stops training if validation accuracy stagnates.

ReduceLROnPlateau: Lowers learning rate when the metric stops improving to find the global minimum.
