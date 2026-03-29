import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
import emoji

# Download stopwords if not present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = emoji.replace_emoji(text, replace="") # Remove emojis
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Dummy dataset for training
data = {
    "text": [
        "I love this new product! It's absolutely amazing",
        "The service was terrible and the staff was rude.",
        "It's an okay experience, nothing special but not bad either.",
        "Best purchase I've made all year! Highly recommend.",
        "I'm so disappointed with the quality. Waste of money.",
        "This app changed my life for the better.",
        "Worst experience ever. Never buying again.",
        "Average product. Does the job.",
        "Fantastic customer support!",
        "The delivery was late and packaging damaged.",
        "I am so happy with this!",
        "Completely useless, do not buy.",
        "It works fine, but could be improved.",
        "Incredible quality and fast shipping.",
        "The product arrived broken and customer service ignored me.",
        "Not bad, not great, just okay.",
        "I will recommend this to all my friends.",
        "A total scam. Avoid at all costs.",
        "Decent for the price.",
        "I absolutely adore this application."
    ],
    "sentiment": [
        "Positive", "Negative", "Neutral", "Positive", "Negative",
        "Positive", "Negative", "Neutral", "Positive", "Negative",
        "Positive", "Negative", "Neutral", "Positive", "Negative",
        "Neutral", "Positive", "Negative", "Neutral", "Positive"
    ]
}

df_dummy = pd.DataFrame(data)

# Synthetically upsample the ENTIRE dummy dataset by 50x
# This balances the dataset against 1000 random Sentiment140 tweets for blazing fast robust training!
df_dummy_upsampled = pd.concat([df_dummy] * 50, ignore_index=True)

# Load Sentiment140 dataset
print("Loading Sentiment140 dataset...")
try:
    cols = ['target', 'id', 'date', 'flag', 'user', 'text']
    df_s140 = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, names=cols)
    
    # Map target values to match dummy dataset formats
    # 0 = negative, 2 = neutral, 4 = positive
    sentiment_map = {0: 'Negative', 2: 'Neutral', 4: 'Positive'}
    df_s140['sentiment'] = df_s140['target'].map(sentiment_map)
    
    # Due to the large size (1.6 million rows), training SVC can take an extremely long time.
    # Sampling 1000 rows to ensure the script completes instantly (<5 seconds)
    SAMPLE_SIZE = 1000
    if len(df_s140) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} rows from Sentiment140 to speed up training...")
        df_s140 = df_s140.sample(n=SAMPLE_SIZE, random_state=42)
        
    df_s140 = df_s140[['text', 'sentiment']]
    
    # Combine datasets
    df = pd.concat([df_dummy_upsampled, df_s140], ignore_index=True)
    print(f"Combined dataset size: {len(df)} rows.")
except FileNotFoundError:
    print("Sentiment140 dataset not found. Using only dummy dataset.")
    df = df_dummy_upsampled

# Preprocessing
df['cleaned'] = df['text'].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['sentiment']

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X, y)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X, y)

# Train SVM
svm_model = SVC(probability=True)
svm_model.fit(X, y)

# Save the models and vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)

with open("lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("Models trained and saved successfully.")
