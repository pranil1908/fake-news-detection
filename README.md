# 📰 Fake News Detection Model

This is a simple machine learning model that detects fake news using Logistic Regression and TF-IDF vectorization.

## 📁 Files

- `fake_news_detector.py` — main model training and evaluation
- `model.pkl` — trained model file
- `dataset.csv` — input data (must have `text` and `label` columns)
- `requirements.txt` — dependencies

## 🚀 How to Run

```bash
pip install -r requirements.txt
python fake_news_detector.py
```

## ✅ Output

The model prints a classification report and saves the model as `model.pkl`.
