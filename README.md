# Artificial Intelligence Internship Projects

Projects developed during my Artificial Intelligence internship, implementing core AI and machine learning concepts on real-world datasets using Python.

---

## Project 1 — Weather Data Analysis and Temperature Prediction

Developed an AI-based temperature prediction system by analyzing historical weather data using regression techniques.

**Objective:**
To apply machine learning algorithms on real weather data to analyze patterns and predict future temperature trends.

**Approach:**
- Performed exploratory data analysis (EDA) on a large historical weather dataset
- Identified key features affecting temperature using correlation analysis
- Built a Linear Regression model for temperature forecasting
- Visualized data trends and model performance through multiple charts

**Libraries used:** pandas, numpy, matplotlib, seaborn, scikit-learn

**Dataset:** [Weather History Dataset - Kaggle](https://www.kaggle.com/datasets/muthuj7/weather-dataset)

**Results:**
- Successfully predicted temperature with low Mean Absolute Error (MAE)
- Identified humidity and pressure as the strongest features correlated with temperature

---

## Project 2 — Spam Email Classifier

Built an AI-powered spam detection system using Natural Language Processing (NLP) to automatically classify messages as spam or legitimate.

**Objective:**
To develop a text classification model using NLP and machine learning that detects spam messages with high accuracy.

**Approach:**
- Preprocessed and cleaned SMS text data for model training
- Applied Bag of Words technique using CountVectorizer for feature extraction
- Trained a Multinomial Naive Bayes classifier on labeled spam/ham data
- Evaluated model using accuracy score and confusion matrix visualization

**Libraries used:** pandas, matplotlib, seaborn, scikit-learn

**Dataset:** [SMS Spam Collection Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

**Results:**
- Achieved high classification accuracy on unseen test data
- Model successfully identifies spam patterns from raw text content

---

## Setup and Usage

**Install dependencies:**
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Run projects:**
```
python weather_prediction.py
python spam_classifier.py
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pandas & numpy | Data manipulation |
| scikit-learn | AI / ML models |
| matplotlib & seaborn | Data visualization |
| NLP (CountVectorizer) | Text feature extraction |

---

*Projects completed as part of an Artificial Intelligence internship, demonstrating hands-on experience in AI, machine learning, data analysis, and NLP.*
