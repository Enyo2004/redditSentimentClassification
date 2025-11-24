Reddit Artist Sentiment Analysis

A Deep Learning system designed to classify Reddit posts about artists into three sentiment categories: Negative, Positive, and Neutral. This project compares the performance of 2 different Neural Network architectures using Keras and TensorFlow.

By applying Natural Language Processing (NLP) techniques, specifically Sequence Modeling and Transfer Learning, the system aims to automate the analysis of public opinion and social sentiment regarding artists.

📋 Table of Contents

About The Project

Project Structure

Built With

Getting Started

Prerequisites

Installation

Usage

1. Data Preparation

2. Training

3. Evaluation

Models Compared

Contributing

License

Contact

📖 About The Project

Analyzing social media sentiment is a task that provides crucial insights into public perception. This project leverages Deep Learning to automate this classification.

Key Features:

Sentiment Classification: Categorizes text into 3 distinct classes (Negative, Positive, Neutral).

Model Comparison: Systematically evaluates a custom Bi-LSTM against a Transfer Learning approach (Universal Sentence Encoder).

Robust Pipeline: Includes dedicated scripts for data loading, visualization, training, and helper functions.

Optimized Performance: Utilizes tf.data.AUTOTUNE and efficient batching for faster training.

📂 Project Structure

The repository is organized into the following directories:

REDDITSENTIMENT/
├── dataset/                           # Scripts for data loading and exploratory data analysis
│   ├── data.py
│   └── explore_data.py
├── Functions/                         # Utility functions for plotting and metrics
│   └── helperFunctions.py
├── model/                             # Custom layer definitions and USE wrapper
│   ├── extra_layers.py
│   └── USE_model.py
├── models/                            # Training scripts and saved model artifacts
│   ├── saved_models/
│   ├── model1.py                      # Bi-LSTM Training Script
│   └── model2.py                      # USE Transfer Learning Training Script
├── reddit_artist_posts_sentiment.csv  # The raw dataset
└── README.md                          # Project documentation


🛠 Built With

Python 3.x

TensorFlow & Keras

TensorFlow Hub

NumPy

Pandas

Matplotlib

🚀 Getting Started

To get a local copy up and running, follow these steps.

Prerequisites

Python 3.6+

pip package manager

Installation

Clone the repository

git clone [https://github.com/your-username/reddit-sentiment-analysis.git](https://github.com/your-username/reddit-sentiment-analysis.git)
cd REDDITSENTIMENT


Install required packages

pip install pandas numpy tensorflow keras tensorflow-hub matplotlib scikit-learn


💻 Usage

1. Data Preparation

The dataset reddit_artist_posts_sentiment.csv is included in the root directory. The dataset/data.py script handles loading, and dataset/explore_data.py can be used to visualize label distribution.

2. Training

Navigate to the models directory. This folder contains scripts for the 2 different models. Run the script corresponding to the model you wish to train.

Train Model 1 (Bi-LSTM):

python models/model1.py


Train Model 2 (Universal Sentence Encoder):

python models/model2.py


Note: The scripts automatically handle environment variables for UTF-8 encoding and TensorFlow logging.

3. Evaluation

Both training scripts include evaluation steps that run automatically after training. They output:

Accuracy Score

Loss Value

Loss Curves Plot (using Functions/helperFunctions.py)

🧠 Models Compared

This project evaluates two distinct architectures to benchmark performance on sentiment classification.

Model 1: Bidirectional LSTM

Uses TextVectorization and an Embedding layer.

Features a Bidirectional LSTM layer to capture sequence context.

Utilizes Global Max Pooling for feature extraction.

Model 2: Universal Sentence Encoder (USE)

Uses Transfer Learning with the Universal Sentence Encoder from TensorFlow Hub.

Implements a custom Keras Layer wrapper.

Features fully connected Dense layers with ReLU activation.
