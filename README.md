# 📊 Social Sentiment Pro

### Real-time Social Media Sentiment Analysis using Machine Learning

---

## 🚀 Project Overview

**Social Sentiment Pro** is a web-based application that analyzes user-generated content such as tweets, comments, and reviews to determine sentiment.

The system classifies input text into:

* ✅ Positive
* ❌ Negative
* 😐 Neutral

It supports:

* Manual text input
* CSV dataset upload
* Multiple ML algorithms (Naive Bayes, Logistic Regression, SVM)

---

## 🎯 Features

* 🔍 Real-time sentiment prediction
* 📂 CSV file upload support
* 🤖 Multiple ML models comparison
* 📊 Accuracy calculation (when labels available)
* 📈 Dashboard visualization (charts)
* 💡 Confidence score display
* 🧠 Text preprocessing (stopwords, punctuation, emoji removal)

---

## 🏗️ Project Architecture (Flow)

1. **User Input**

   * Enter text OR upload CSV

2. **Data Preprocessing**

   * Lowercase conversion
   * Remove stopwords
   * Remove punctuation & emojis

3. **Feature Extraction**

   * TF-IDF Vectorization

4. **Model Prediction**

   * Naive Bayes
   * Logistic Regression
   * Support Vector Machine (SVM)

5. **Output**

   * Sentiment (Positive / Negative / Neutral)
   * Confidence Score
   * Accuracy (if dataset has labels)

6. **Visualization**

   * Pie Chart (Sentiment Distribution)
   * Bar Chart (Model Accuracy)

---

## 🧠 Machine Learning Models Used

| Algorithm           | Description                              |
| ------------------- | ---------------------------------------- |
| Naive Bayes         | Fast and simple probabilistic classifier |
| Logistic Regression | Best for text classification             |
| SVM                 | High accuracy classifier for NLP         |

---

## 📂 Dataset

* **Sentiment140 Dataset (1.6M tweets)** used for training
* Labels:

  * 0 → Negative
  * 4 → Positive
* Neutral class added manually

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/social-sentiment-pro.git
cd social-sentiment-pro
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Models

```bash
python train_models.py
```

### 4️⃣ Run Application

```bash
python app.py
```

### 5️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 📄 CSV Format for Accuracy

To calculate **real accuracy**, CSV must include labels:

```csv
text,sentiment
I love this product,Positive
Worst service ever,Negative
It is okay,Neutral
```

---

## ⚠️ Important Notes

* Accuracy is calculated **only when true labels are available**
* For manual input, **confidence score is shown instead of accuracy**
* Model performance depends on training dataset quality

---

## 📊 Output Example

| Algorithm           | Positive | Negative | Neutral | Accuracy |
| ------------------- | -------- | -------- | ------- | -------- |
| Naive Bayes         | 4        | 4        | 2       | 80%      |
| Logistic Regression | 5        | 3        | 2       | 88%      |
| SVM                 | 4        | 4        | 2       | 90%      |

---

## 🛠️ Technologies Used

* Frontend: HTML, CSS, JavaScript (Tailwind, Chart.js)
* Backend: Python (Flask)
* Machine Learning: Scikit-learn
* NLP: NLTK

---

## 📌 Future Enhancements

* 🔹 Deep Learning (LSTM / BERT)
* 🔹 Real-time Twitter API integration
* 🔹 WordCloud visualization
* 🔹 Confusion Matrix & F1 Score
* 🔹 Export results as CSV

---

## 👨‍💻 Author

**Niyaz Pathan**
BCA Final Year Project

---

## 🙏 Acknowledgment

This project was developed with guidance from **Caliber Institute of Technology, Hingna, Nagpur**.

---

## 📜 License

This project is for academic and educational purposes.
