🧠 Sentiment Analysis using NLTK & Machine Learning
🎯 Project Overview

This project focuses on Sentiment Analysis, a Natural Language Processing (NLP) technique used to classify text as positive, negative, or neutral.
Using the Sentiment140 dataset, the model analyzes and predicts emotions in text data such as tweets or product reviews.

The project demonstrates a complete end-to-end machine learning pipeline — from data preprocessing and feature extraction to model training, evaluation, and visualization.

🚀 Key Features

🧹 Text Preprocessing: Cleans raw text by removing URLs, mentions, hashtags, and punctuation.

🔤 Lemmatization: Converts words into their base form using NLTK’s WordNetLemmatizer.

📊 Feature Extraction: Uses TF-IDF vectorization to transform text into numerical vectors.

🤖 Model Training: Logistic Regression classifier for accurate sentiment classification.

📈 Evaluation Metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

💬 Custom Predictions: Predicts sentiment for any user-inputted text.

📉 Visualizations: Includes sentiment distribution and confusion matrix heatmaps.

🧰 Technologies Used

Programming Language: Python

Libraries:

NLTK – Natural Language Processing

scikit-learn – Machine Learning Model & Evaluation

pandas, numpy – Data Handling

matplotlib, seaborn – Data Visualization

📂 Dataset

Dataset: Sentiment140 (Kaggle)
Contains 1.6 million tweets labeled as:

0 → Negative

2 → Neutral

4 → Positive

Only the text and sentiment columns are used in this project.

⚙️ How It Works

Data Cleaning: Remove noise, links, and special symbols.

Text Preprocessing: Tokenize, remove stopwords, and lemmatize text.

Feature Engineering: Apply TF-IDF vectorization.

Model Training: Fit Logistic Regression on the processed data.

Evaluation: Generate accuracy reports and confusion matrix visualizations.

Prediction: Classify any custom text input as positive, negative, or neutral.

📊 Model Performance

Accuracy: ~85%

Precision/Recall: High reliability for positive and negative classes.

Confusion Matrix: Shows strong class separation and minimal misclassification.

🧪 Example Predictions
Input Text	Predicted Sentiment
I love this product!	Positive
This is the worst experience ever.	Negative
The movie was okay, not great but not bad.	Neutral
📦 Installation
# Clone this repository
git clone https://github.com/yourusername/sentiment-analysis-nltk.git

# Navigate to the folder
cd sentiment-analysis-nltk

# Install dependencies
pip install -r requirements.txt

▶️ Usage
python sentiment_analysis_project.py


To test your own text:

predict_sentiment("I am feeling great today!")

🧭 Future Improvements

Integrate Deep Learning (LSTM, BERT) models for improved context understanding.

Deploy as a web app using Flask or Streamlit.

Extend dataset for domain-specific sentiment analysis (e.g., product reviews, movie ratings).

👨‍💻 Author

Rahil Khan
Guided by curiosity, driven by growth — turning data into insights, code into solutions, and every challenge into an opportunity to learn.
🔗 LinkedIn Profile: https://www.linkedin.com/in/rahil-khan-06a653297/
