
# 📧 Spam Detection with Machine Learning

This project uses machine learning techniques to classify emails as **spam** or **not spam** based on their content. The dataset is cleaned, visualized, and processed to train a model that can effectively identify spam messages.

## 📂 Dataset
The dataset used is `spam.csv` which includes email messages labeled as `spam` or `ham`. It is preprocessed to remove unnecessary columns and clean text data.

## Features
- Text preprocessing (removal of stopwords, punctuation, etc.)
- Data visualization with Seaborn and Matplotlib
- Feature extraction using TF-IDF vectorization
- Model training using Naive Bayes classifier
- Evaluation with accuracy, confusion matrix, and classification report


## 📊 Results
The model was trained using the **Multinomial Naive Bayes** algorithm and achieved high accuracy in classifying emails. Evaluation metrics include:
- 97.1% Accuracy score
- 100% Precision (1)

## Tools and Libraries
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

