🎨 Reddit Artist Sentiment Analysis
📖 Overview
This project is a Deep Learning application designed to perform sentiment analysis on Reddit posts related to artists. It classifies text data into three distinct categories: Negative, Positive, and Neutral.
The repository demonstrates a comparative study between two different Natural Language Processing (NLP) approaches using TensorFlow and Keras:
Custom Sequence Modeling: A Bidirectional LSTM network trained from scratch.
Transfer Learning: A model utilizing the Universal Sentence Encoder (USE) from TensorFlow Hub.
📂 Project Structure
The project is organized into a modular directory structure for better maintainability:
REDDITSENTIMENT/
├── dataset/
│   ├── data.py              # Scripts for loading and processing the CSV data
│   └── explore_data.py      # Exploratory Data Analysis (EDA) and visualization
├── Functions/
│   └── helperFunctions.py   # Utility functions (e.g., plotting loss/accuracy curves)
├── model/
│   ├── extra_layers.py      # Definitions for custom Keras layers
│   └── USE_model.py         # Wrapper class for the Universal Sentence Encoder
├── models/
│   ├── saved_models/        # Directory where trained models (.keras) and weights are saved
│   ├── model1.py            # Training and evaluation script for the Bi-LSTM model
│   └── model2.py            # Training and evaluation script for the USE Transfer Learning model
└── reddit_artist_posts_sentiment.csv  # The source dataset


📊 The Dataset
The model is trained on the reddit_artist_posts_sentiment.csv file.
Input (X): Raw text from Reddit posts.
Target (y): Sentiment labels converted to numeric values.
Label Encoding:
| Sentiment | Class ID |
| :--- | :--- |
| Negative | 0 |
| Positive | 1 |
| Neutral | 2 |
🧠 Model Architectures
Model 1: Bidirectional LSTM
This model processes text as a sequence of tokens.
Text Vectorization: Adapts to the training set vocabulary (max tokens: 1000).
Embedding: Maps tokens to a 256-dimensional vector space.
Bi-LSTM Layer: A Bidirectional Long Short-Term Memory layer (64 units) to capture context from both directions.
Pooling: Global Max Pooling 1D to extract the most salient features.
Output: Dense layer with Softmax activation for multiclass classification.
Model 2: Universal Sentence Encoder (USE)
This model utilizes Transfer Learning for robust sentence embeddings.
Base: Pre-trained Universal Sentence Encoder (from TensorFlow Hub).
Hidden Layers: Two Dense layers with 128 units each and ReLU activation.
Output: Dense layer with Softmax activation.
⚙️ Technical Features
Data Pipeline: Uses tf.data.Dataset with batch() and prefetch(tf.data.AUTOTUNE) for optimized memory usage and training speed.
Callbacks:
EarlyStopping: Monitors validation accuracy to prevent overfitting.
ReduceLROnPlateau: Dynamically adjusts the learning rate if convergence stalls.
Environment Handling: Automatically sets TF_CPP_MIN_LOG_LEVEL and PYTHONUTF8 to ensure smooth execution.
🚀 Getting Started
Prerequisites
Ensure you have Python 3.x installed. Install the required dependencies:
pip install pandas numpy tensorflow keras tensorflow-hub matplotlib scikit-learn


Usage
Clone the Repository:
git clone [https://github.com/your-username/reddit-sentiment-analysis.git](https://github.com/your-username/reddit-sentiment-analysis.git)
cd REDDITSENTIMENT


Train Model 1 (Bi-LSTM):
python models/model1.py


This will preprocess the data, train the LSTM network, save the model to models/saved_models/model1/, and display the loss curves.
Train Model 2 (USE):
python models/model2.py


This will download the USE model, train the dense layers, save the model to models/saved_models/model2/, and display the evaluation metrics.
📈 Results
Both models output the following metrics upon completion:
Loss: Sparse Categorical Crossentropy.
Accuracy: Percentage of correctly classified posts.
Loss curves are automatically plotted using matplotlib at the end of the training script.
